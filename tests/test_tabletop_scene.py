import numpy as np

from scripts import tabletop_scene as ts


def table(rng, count=60000):
    xy = rng.uniform((0.15, -0.6), (0.9, 0.6), (count, 2))
    return np.column_stack((xy, 0.75 + rng.normal(0, 0.003, count)))


def can(rng, centre, height=0.115, radius=0.033):
    """What depth sees of a can: its lid, the side facing the robot, and a low skirt."""
    angle, distance = rng.uniform(0, 2 * np.pi, 400), radius * np.sqrt(rng.uniform(0, 1, 400))
    lid = np.column_stack((centre[0] + distance * np.cos(angle), centre[1] + distance * np.sin(angle),
                           np.full(400, 0.75 + height)))
    facing = rng.uniform(np.pi / 2, 3 * np.pi / 2, 400)
    side = np.column_stack((centre[0] + radius * np.cos(facing), centre[1] + radius * np.sin(facing),
                            0.75 + rng.uniform(0, height, 400)))
    skirt = np.column_stack((centre[0] + rng.uniform(-0.09, 0.04, 300),
                             centre[1] + rng.uniform(-0.07, 0.07, 300),
                             0.75 + rng.uniform(0.02, 0.04, 300)))
    return np.vstack((lid, side, skirt))


def box(rng, centre, half=0.15, height=0.07, count=6000):
    along = rng.uniform(-half, half, count)
    wall = rng.integers(0, 4, count)
    x = centre[0] + np.where(wall < 2, along, np.where(wall == 2, -half, half))
    y = centre[1] + np.where(wall < 2, np.where(wall == 0, -half, half), along)
    return np.column_stack((x, y, 0.75 + rng.uniform(0, height, count)))


def scene(*things, seed=0):
    rng = np.random.default_rng(seed)
    arm = np.vstack([table(rng)] + [make(rng, *args) for make, *args in things])
    arm = arm + rng.normal(0, 0.001, arm.shape)
    return arm, ts.fit_table_plane(arm)


def cans_in(objects):
    return sorted((round(o.center[0], 2), round(o.center[1], 2))
                  for o in ts.graspable_candidates(objects))


def test_two_cans_joined_by_depth_skirt_are_two_objects():
    arm, plane = scene((can, (0.30, -0.20)), (can, (0.30, -0.29)))

    assert cans_in(ts.find_objects(arm, plane)) == [(0.30, -0.29), (0.30, -0.20)]


def test_a_can_is_measured_by_its_body_not_its_skirt():
    arm, plane = scene((can, (0.35, 0.10)))

    found = ts.select_graspable(ts.find_objects(arm, plane))

    assert np.allclose(found.center[:2], (0.35, 0.10), atol=0.006)
    assert 0.055 <= found.length <= 0.075 and 0.055 <= found.width <= 0.075
    assert abs(found.top - 0.115) < 0.01


def test_cans_leaning_on_the_box_are_freed_and_the_box_stays_one_object():
    arm, plane = scene((box, (0.40, 0.0)), (can, (0.30, -0.195)), (can, (0.45, -0.195)))

    objects = ts.find_objects(arm, plane)

    found = sorted(o.center[:2] for o in ts.graspable_candidates(objects))
    assert np.allclose(found, [(0.30, -0.195), (0.45, -0.195)], atol=0.01)
    containers = [o for o in objects if o.length > 0.2]
    assert len(containers) == 1
    assert abs(containers[0].length - 0.30) < 0.03 and abs(containers[0].width - 0.30) < 0.03
    assert ts.find_box(arm, plane, objects, exclude=ts.select_graspable(objects)) is containers[0]


def test_a_box_with_an_uneven_rim_is_not_broken_into_graspable_pieces():
    rng = np.random.default_rng(3)
    walls = box(rng, (0.40, 0.0))
    far = walls[:, 0] > 0.54
    walls[far, 2] = 0.75 + (walls[far, 2] - 0.75) * 0.095 / 0.07     # far wall reads taller
    arm = np.vstack((table(rng), walls))
    plane = ts.fit_table_plane(arm)

    objects = ts.find_objects(arm, plane)

    assert cans_in(objects) == []
    assert len([o for o in objects if o.length > 0.2]) == 1
