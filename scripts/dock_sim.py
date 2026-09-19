"""Table docking in the sim: start the robot somewhere in front of the fold table, let it find the table
with its head camera and drive square up to it, then measure where it stopped.

    uv run --locked --extra yolo python scripts/dock_sim.py --trials 10            # numbers only
    uv run --locked --extra yolo python scripts/dock_sim.py --trials 3 --view      # watch it (needs a desktop)

The table mask here is the renderer's segmentation (perfect perception), so this measures the geometry
and control. On the robot the same code gets its mask from YOLO-seg class "table".
Target: the fold pose, i.e. the base where the sim's fold scene puts it (table edge 11 cm ahead, square).
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bbsim.dock import DockController, edge_from_mask  # noqa: E402
from bbsim.headcam import HeadCamera  # noqa: E402
from bbsim.origami_scene import OrigamiScene  # noqa: E402

DOCK_ROI = (160, 300, 1120, 960)  # the middle and bottom of the left eye: where a table 0.3-1 m ahead appears


def yaw_of(data, body):
    r = data.xmat[body].reshape(3, 3)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def run(seed, view=False, max_seconds=45., hold=(.5, .2, .08), rate=10):
    rng = np.random.default_rng(seed)
    cam = HeadCamera(roi=DOCK_ROI)
    scene = OrigamiScene("fox", seed, None, [cam])
    cam.attach(scene.model)
    m, d, sim = scene.model, scene.data, scene.sim
    base, table = sim.base, scene.table_geom
    table_top = float(m.geom_pos[table][2] + m.geom_size[table][2])
    near_edge_x = float(m.geom_pos[table][0] - m.geom_size[table][0])
    target_x = near_edge_x - .11  # where the fold scene has the base: edge 11 cm ahead of the axle
    # Start pose: back from the table, off to a side, turned.
    start = np.array([target_x - rng.uniform(.3, .9), rng.uniform(-.15, .15)])
    yaw0 = rng.uniform(-np.deg2rad(25), np.deg2rad(25))
    d.qpos[:2] = start
    d.qpos[3:7] = [np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]
    d.qvel[:] = 0
    sim.arm_targets[:] = 0
    mujoco.mj_forward(m, d)
    controller = DockController(hold=hold, dt=1 / rate)
    odometry, last_xy = 0., d.xpos[base][:2].copy()
    trace, max_force = [], 0.
    window = None
    if view:
        import cv2
        window = "Docking (sim head camera)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while d.time < max_seconds:
        heading = yaw_of(d, base)
        ids = cam.segmentation(d)
        mask = cam.top_face(d, ids == table, m, table)  # the tabletop only: the front face would pull the edge ~1 cm closer

        def pixel_to_base(pixel, z):
            p = cam.to_plane(d, pixel, z) - d.xpos[base]
            c, s = np.cos(heading), np.sin(heading)
            return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2]])

        edge = edge_from_mask(mask, pixel_to_base, table_top, rng)
        velocity = float(sim.state()[3])  # forward speed: on the robot, from wheel odometry
        v, w, phase = controller.step(edge, odometry, heading, velocity)
        sim.command[:] = [v, w]
        trace.append(dict(t=round(float(d.time), 2), phase=phase, v=round(v, 3), w=round(w, 3), true_mm=round((near_edge_x - float(d.xpos[base][0])) * 1000, 1),
                          edge=None if edge is None else [round(edge.distance, 4), round(float(np.degrees(edge.yaw)), 2)]))
        if window:
            import cv2
            image = np.ascontiguousarray(cam.render(d)[:, :, ::-1])
            image[mask] = (image[mask] * .6 + np.array([0, 120, 0]) * .4).astype(np.uint8)
            label = f"t {d.time:4.1f}s  {phase}  v {v:+.3f}  w {w:+.2f}" + ("" if edge is None else f"  edge {edge.distance * 100:5.1f} cm  yaw {np.degrees(edge.yaw):+5.1f} deg")
            cv2.putText(image, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if phase == "DOCKED" and not window:
            break
        for _ in range(int(200 / rate)):  # one control period, arms held with gravity compensation like the fold scene
            d.qfrc_applied[scene.arm_dof] = d.qfrc_bias[scene.arm_dof] - d.qfrc_passive[scene.arm_dof]
            sim.step()
            xy = d.xpos[base][:2].copy()
            h = yaw_of(d, base)
            odometry += float((xy - last_xy) @ [np.cos(h), np.sin(h)])  # what wheel odometry measures
            last_xy = xy
            for i in range(d.ncon):
                con = d.contact[i]
                if table in (con.geom1, con.geom2):
                    f = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, f)
                    max_force = max(max_force, float(np.linalg.norm(f[:3])))
            if sim.failed:
                break
        if sim.failed:
            break
    # Docked: command zero for 2 s (on the robot LEAN would engage here), then score where it is.
    sim.command[:] = 0
    for _ in range(400):
        sim.step()
    final = d.xpos[base][:2]
    result = dict(seed=seed, start=[round(float(v), 3) for v in start], start_yaw_deg=round(float(np.degrees(yaw0)), 1),
                  docked=controller.phase == "DOCKED", fell=sim.failed, seconds=round(float(d.time), 1),
                  forward_error_mm=round(float(final[0] - target_x) * 1000, 1), lateral_mm=round(float(final[1]) * 1000, 1),
                  yaw_error_deg=round(float(np.degrees(yaw_of(d, base))), 2), max_table_contact_n=round(max_force, 2))
    cam.close()
    scene.close()
    if window:
        import cv2
        cv2.destroyWindow(window)
    return result, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--first-seed", type=int, default=100)
    p.add_argument("--view", action="store_true")
    p.add_argument("--out", type=Path, default=Path("artifacts/dock"))
    p.add_argument("--hold", type=float, nargs=3, default=(.5, .2, .08), help="kp kd limit of the final position hold")
    p.add_argument("--rate", type=int, default=10, help="control rate, Hz")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(a.first_seed, a.first_seed + a.trials):
        result, trace = run(seed, a.view, hold=tuple(a.hold), rate=a.rate)
        results.append(result)
        print(json.dumps(result), flush=True)
        (a.out / f"trace_{seed}.json").write_text(json.dumps(trace) + "\n")
    ok = [r for r in results if r["docked"] and not r["fell"]]
    summary = dict(trials=len(results), docked=len(ok),
                   forward_error_mm=dict(mean=round(float(np.mean([abs(r["forward_error_mm"]) for r in ok])), 1) if ok else None,
                                         worst=max((abs(r["forward_error_mm"]) for r in ok), default=None)),
                   yaw_error_deg=dict(mean=round(float(np.mean([abs(r["yaw_error_deg"]) for r in ok])), 2) if ok else None,
                                      worst=max((abs(r["yaw_error_deg"]) for r in ok), default=None)),
                   table_contact_n_worst=max((r["max_table_contact_n"] for r in results), default=None))
    (a.out / "summary.json").write_text(json.dumps(dict(summary=summary, results=results), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
