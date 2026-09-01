import unittest

import numpy as np

from src.algorithms.bosco import BoscoExpert


class BoscoCollisionShieldTest(unittest.TestCase):
    def setUp(self):
        self.expert = BoscoExpert.__new__(BoscoExpert)
        self.expert.n = 2
        self.expert.dt = 0.1
        self.expert.robot_radius = 0.2
        self.expert.collision_margin = 0.02

    def test_stops_a_robot_before_its_next_pose_overlaps_a_teammate(self):
        positions = np.array([[0.0, 0.0], [0.45, 0.0]], dtype=np.float32)
        headings = np.zeros(2, dtype=np.float32)
        velocities = np.array([1.0, 0.0], dtype=np.float32)
        omegas = np.zeros(2, dtype=np.float32)

        safe_velocities = self.expert._collision_shield(
            positions, headings, velocities, omegas
        )
        next_positions = self.expert._predict_positions(
            positions, headings, safe_velocities, omegas
        )

        self.assertEqual(float(safe_velocities[0]), 0.0)
        self.assertGreaterEqual(
            float(np.linalg.norm(next_positions[0] - next_positions[1])),
            2.0 * self.expert.robot_radius + self.expert.collision_margin,
        )


if __name__ == '__main__':
    unittest.main()
