"""Curriculum controlling how strongly humans react to robots during training."""

from __future__ import annotations

import jax.numpy as jnp


def ghost_robot_probability(collision_rate, coverage_rate, collision_target: float = 0.1):
    """Return 1 at zero progress and decay monotonically toward 0.

    Collision progress saturates at ``collision_target``; coverage progress is
    already a fraction.  Multiplication makes either learned signal sufficient
    to phase in human reactions while ensuring both can only lower ghosting.
    """
    collision_progress = jnp.clip(collision_rate / collision_target, 0.0, 1.0)
    coverage_progress = jnp.clip(coverage_rate, 0.0, 1.0)
    return (1.0 - collision_progress) * (1.0 - coverage_progress)
