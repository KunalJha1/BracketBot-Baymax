import math

import numpy as np
import pytest

from follow_perception import ClusterConfig, base_to_local, find_people

RNG = np.random.default_rng(7)


def standing_person(forward, left=0.0, height=1.75, radius=0.18, n=900):
    """Points on the half of a vertical cylinder that faces the robot (what stereo sees)."""
    centre = np.array([forward, left])
    toward_robot = -centre / np.linalg.norm(centre)
    side = np.array([-toward_robot[1], toward_robot[0]])
    phi = RNG.uniform(-math.pi / 2, math.pi / 2, n)
    xy = centre + radius * (np.cos(phi)[:, None] * toward_robot + np.sin(phi)[:, None] * side)
    z = RNG.uniform(0.05, height, n)
    return np.column_stack([xy, z])


def floor(n=2000):
    return np.column_stack([RNG.uniform(0.3, 3.5, n), RNG.uniform(-2.0, 2.0, n), RNG.normal(0.0, 0.01, n)])


def slab(f0, f1, l0, l1, z0, z1, n):
    return np.column_stack([RNG.uniform(f0, f1, n), RNG.uniform(l0, l1, n), RNG.uniform(z0, z1, n)])


def test_a_standing_person_is_found_at_the_front_of_their_torso():
    people = find_people(np.vstack([standing_person(1.2, 0.3), floor()]))
    assert len(people) == 1
    person = people[0]
    # the camera sees the front surface: about 0.7 of the radius nearer than the centre
    assert person.forward == pytest.approx(1.2 - 0.7 * 0.18, abs=0.05)
    assert person.left == pytest.approx(0.3 - 0.3 / 1.24 * 0.7 * 0.18, abs=0.05)
    assert person.top >= 1.6


def test_floor_table_wall_and_box_are_not_people():
    table = np.vstack([slab(1.0, 1.8, -0.4, 0.4, 0.72, 0.76, 1500),  # top
                       slab(1.0, 1.05, -0.4, -0.35, 0.1, 0.72, 100), slab(1.75, 1.8, 0.35, 0.4, 0.1, 0.72, 100)])
    wall = slab(2.95, 3.0, -2.0, 2.0, 0.1, 2.0, 4000)
    box = slab(0.8, 1.1, -0.9, -0.6, 0.1, 0.4, 600)
    assert find_people(np.vstack([floor(), table, wall, box])) == []


def test_two_separated_people_give_two_clusters_nearest_first():
    people = find_people(np.vstack([standing_person(2.0, 0.6), standing_person(1.3, -0.4)]))
    assert len(people) == 2
    assert people[0].forward < people[1].forward
    assert people[0].left < 0 < people[1].left


def test_a_person_a_step_away_from_a_wall_is_still_found():
    wall = slab(1.75, 1.8, -2.0, 2.0, 0.1, 2.0, 4000)
    people = find_people(np.vstack([standing_person(1.3), wall]))
    assert len(people) == 1
    assert people[0].forward == pytest.approx(1.3 - 0.7 * 0.18, abs=0.05)


def test_a_person_against_a_wall_merges_with_it_and_is_lost():
    # documented limit: within about 10 cm the person and wall form one too-wide cluster
    wall = slab(1.33, 1.38, -2.0, 2.0, 0.1, 2.0, 4000)
    assert find_people(np.vstack([standing_person(1.3), wall])) == []


def test_crop_and_empty_input():
    assert find_people(standing_person(4.0)) == []
    assert find_people(np.empty((0, 3))) == []
    assert find_people(standing_person(1.2, n=40)) == []  # too few points
    assert find_people(standing_person(1.2), ClusterConfig(min_top=1.9)) == []


def test_base_frame_points_become_forward_left_up():
    local = base_to_local(np.array([[0.2, 1.0, 0.5]]))
    assert local[0] == pytest.approx([1.0, -0.2, 0.5])
