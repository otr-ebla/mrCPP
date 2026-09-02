import jax.numpy as jnp

from src.algorithms.jax_bosco import JaxGuideState, jax_guide_step


def _line_neighbors(n):
    return jnp.asarray([
        [cell - 1 if cell > 0 else -1,
         cell + 1 if cell + 1 < n else -1,
         -1,
         -1]
        for cell in range(n)
    ], dtype=jnp.int32)


def test_finished_tour_replans_towards_uncovered_cell():
    tours = jnp.full((1, 1, 4), -1, dtype=jnp.int32).at[0, 0, 0].set(0)
    state = JaxGuideState(
        target=jnp.asarray([[1]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=tours,
        tour_lens=jnp.asarray([[1]], dtype=jnp.int32),
    )

    new_state, waypoint, reached = jax_guide_step(
        state=state,
        positions=jnp.asarray([[[0.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=jnp.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32),
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(4),
        w=4,
        h=1,
        cell_size=1.0,
        guides=[object()],
        free_cells=jnp.ones((4,), dtype=jnp.bool_),
        graph_components=jnp.zeros((4,), dtype=jnp.int32),
    )

    assert waypoint.tolist() == [[1]]
    assert new_state.target.tolist() == [[1]]
    assert reached.tolist() == [[False]]


def test_finished_tour_stops_when_every_free_cell_is_covered():
    tours = jnp.full((1, 1, 4), -1, dtype=jnp.int32).at[0, 0, 0].set(0)
    state = JaxGuideState(
        target=jnp.asarray([[1]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=tours,
        tour_lens=jnp.asarray([[1]], dtype=jnp.int32),
    )

    _, waypoint, _ = jax_guide_step(
        state=state,
        positions=jnp.asarray([[[0.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=jnp.ones((1, 1, 4), dtype=jnp.float32),
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(4),
        w=4,
        h=1,
        cell_size=1.0,
        guides=[object()],
        free_cells=jnp.ones((4,), dtype=jnp.bool_),
        graph_components=jnp.zeros((4,), dtype=jnp.int32),
    )

    assert waypoint.tolist() == [[-1]]


def test_mopup_ignores_closer_uncovered_cell_in_an_unreachable_component():
    tours = jnp.full((1, 1, 5), -1, dtype=jnp.int32).at[0, 0, 0].set(0)
    state = JaxGuideState(
        target=jnp.asarray([[1]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=tours,
        tour_lens=jnp.asarray([[1]], dtype=jnp.int32),
    )
    # Cells 1 and 2 are geometrically close but disconnected.  Cell 4 is the
    # only pending target in the robot's component, reached through cell 3.
    neighbors = jnp.asarray([
        [3, -1, -1, -1],
        [2, -1, -1, -1],
        [1, -1, -1, -1],
        [0, 4, -1, -1],
        [3, -1, -1, -1],
    ], dtype=jnp.int32)

    _, waypoint, _ = jax_guide_step(
        state=state,
        positions=jnp.asarray([[[0.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=jnp.asarray([[[1.0, 0.0, 0.0, 1.0, 0.0]]], dtype=jnp.float32),
        done=jnp.asarray([False]),
        graph_neighbors=neighbors,
        w=5,
        h=1,
        cell_size=1.0,
        guides=[object()],
        free_cells=jnp.ones((5,), dtype=jnp.bool_),
        graph_components=jnp.asarray([0, 1, 1, 0, 0], dtype=jnp.int32),
    )

    assert waypoint.tolist() == [[3]]
