import jax
import jax.numpy as jnp

from src.algorithms.jax_bosco import JaxGuideState, advance_cursors, jax_guide_step


def _line_neighbors(n):
    return jnp.asarray([
        [cell - 1 if cell else -1, cell + 1 if cell + 1 < n else -1, -1, -1]
        for cell in range(n)
    ], dtype=jnp.int32)


def test_advance_cursors_batches_envs_and_robots():
    tours = jnp.asarray([
        [[0, 1, 2, 3, -1], [3, 2, 1, 0, -1]],
        [[0, 1, 1, 2, 3], [4, 3, 2, 1, 0]],
    ], dtype=jnp.int32)
    lens = jnp.asarray([[4, 4], [5, 5]], dtype=jnp.int32)
    idx = jnp.asarray([[0, 0], [1, 0]], dtype=jnp.int32)
    cell = jnp.asarray([[0, 3], [1, 4]], dtype=jnp.int32)
    covered = jnp.asarray([
        [True, True, False, False, False],
        [False, True, True, False, False],
    ])

    actual = jax.jit(advance_cursors)(tours, lens, idx, cell, covered)

    assert actual.tolist() == [[2, 1], [4, 1]]


def test_done_reset_stays_inside_jit():
    state = JaxGuideState(
        target=jnp.asarray([[2]], jnp.int32),
        prev_cell=jnp.asarray([[1]], jnp.int32),
        fail_cov=jnp.asarray([[3]], jnp.int32),
        idx=jnp.asarray([[2]], jnp.int32),
        tours=jnp.asarray([[[0, 1, 2, 3]]], jnp.int32),
        tour_lens=jnp.asarray([[4]], jnp.int32),
    )

    def step(s, done):
        return jax_guide_step(
            s,
            jnp.asarray([[[0.5, 0.5]]], jnp.float32),
            jnp.asarray([[[1.0, 0.0, 0.0, 0.0]]], jnp.float32),
            done,
            _line_neighbors(4),
            4,
            1,
            1.0,
            free_cells=jnp.ones((4,), jnp.bool_),
            graph_components=jnp.zeros((4,), jnp.int32),
        )

    new_state, waypoint, reached = jax.jit(step)(state, jnp.asarray([True]))

    assert new_state.idx.tolist() == [[1]]
    assert waypoint.tolist() == [[1]]
    assert reached.tolist() == [[False]]
