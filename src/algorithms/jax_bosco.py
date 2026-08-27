import jax
import jax.numpy as jnp
from typing import NamedTuple

class JaxGuideState(NamedTuple):
    target: jax.Array        # (E, N)
    prev_cell: jax.Array     # (E, N)
    fail_cov: jax.Array      # (E, N)
    idx: jax.Array           # (E, N)
    tours: jax.Array         # (E, N, MAX_TOUR_LEN)
    tour_lens: jax.Array     # (E, N)

def jax_bellman_ford_to_target(targets, cost, neighbors, n_cells):
    """
    Computes shortest path distances from all cells to the `targets`.
    targets: (E, N)
    cost: (E, C)
    neighbors: (C, 4)
    Returns dist: (E, N, C)
    """
    E, N = targets.shape
    C = n_cells
    
    # Pad cost to handle -1 neighbors
    padded_cost = jnp.pad(cost, ((0, 0), (0, 1)), constant_values=jnp.inf) # (E, C+1)
    
    def init_fn():
        dist = jnp.full((E, N, C + 1), jnp.inf)
        e_idx = jnp.arange(E)[:, None]
        n_idx = jnp.arange(N)[None, :]
        
        # Valid targets are >= 0
        valid = targets >= 0
        safe_targets = jnp.maximum(targets, 0)
        
        dist = jnp.where(valid[..., None], dist.at[e_idx, n_idx, safe_targets].set(0.0), dist)
        return dist
        
    def step_fn(dist):
        padded_neighbors = jnp.where(neighbors >= 0, neighbors, C)
        
        # To compute D(c) = min_{nb} (cost[nb] + D(nb))
        # For each cell c, its neighbors are padded_neighbors[c]
        # We need dist of neighbors: dist[:, :, padded_neighbors] -> (E, N, C, 4)
        # We need cost of neighbors: padded_cost[:, padded_neighbors] -> (E, C, 4)
        
        d_nb = dist[:, :, padded_neighbors] # (E, N, C, 4)
        c_nb = padded_cost[:, None, padded_neighbors] # (E, 1, C, 4)
        
        d_from_c_via_nb = c_nb + d_nb
        
        best_d = jnp.min(d_from_c_via_nb, axis=-1) # (E, N, C)
        
        current_d = dist[:, :, :-1]
        improved = best_d < current_d
        new_dist = jnp.where(improved, best_d, current_d)
        
        next_dist = dist.at[:, :, :-1].set(new_dist)
        changed = jnp.any(improved)
        
        return next_dist, changed

    def cond_fn(carry):
        _, changed = carry
        return changed
        
    def body_fn(carry):
        dist, _ = carry
        return step_fn(dist)
        
    initial_dist = init_fn()
    final_dist, _ = jax.lax.while_loop(cond_fn, body_fn, (initial_dist, jnp.bool_(True)))
    
    return final_dist[:, :, :-1]


def get_next_waypoint(cell, target_dists, cost, neighbors, n_cells):
    """
    Given the distance field `target_dists` (E, N, C) and current `cell` (E, N),
    find the best neighbor to step into.
    """
    E, N = cell.shape
    C = n_cells
    
    padded_cost = jnp.pad(cost, ((0, 0), (0, 1)), constant_values=jnp.inf)
    padded_dist = jnp.pad(target_dists, ((0, 0), (0, 0), (0, 1)), constant_values=jnp.inf)
    padded_neighbors = jnp.where(neighbors >= 0, neighbors, C)
    
    e_idx = jnp.arange(E)[:, None, None]
    n_idx = jnp.arange(N)[None, :, None]
    
    # Neighbors of current cell: (E, N, 4)
    c_neighbors = padded_neighbors[cell] 
    
    # Cost to enter those neighbors: (E, N, 4)
    # cost is (E, C+1). We need cost[e, c_neighbors[e, n, i]]
    c_nb_cost = padded_cost[jnp.arange(E)[:, None, None], c_neighbors]
    
    # Distance from those neighbors to target: (E, N, 4)
    c_nb_dist = padded_dist[e_idx, n_idx, c_neighbors]
    
    total_cost = c_nb_cost + c_nb_dist
    
    best_idx = jnp.argmin(total_cost, axis=-1) # (E, N)
    
    best_neighbor = jnp.take_along_axis(c_neighbors, best_idx[..., None], axis=-1).squeeze(-1)
    
    # If cell is already target, return target
    best_neighbor = jnp.where(target_dists[jnp.arange(E)[:, None], jnp.arange(N)[None, :], cell] == 0.0, cell, best_neighbor)
    
    # If unreachable (all inf), return -1
    best_neighbor = jnp.where(jnp.min(total_cost, axis=-1) == jnp.inf, -1, best_neighbor)
    
    return best_neighbor


def advance_cursors(tours, tour_lens, idx, cell, covered, snap_window=8):
    """
    Advances the cursor for all robots.
    tours: (E, N, MAX_TOUR_LEN)
    tour_lens: (E, N)
    idx: (E, N)
    cell: (E, N)
    covered: (E, C)
    """
    E, N = idx.shape
    
    def advance_single(t, t_len, i, c, cov):
        # t: (MAX_TOUR_LEN,), t_len: int, i: int, c: int, cov: (C,)
        
        # 1. Snap window
        def snap_body(val):
            k, best_k = val
            is_match = (k < t_len) & (t[k] == c)
            new_best = jnp.where(is_match, k + 1, best_k)
            return (k + 1, new_best)
            
        _, i = jax.lax.while_loop(
            lambda val: val[0] < i + snap_window,
            snap_body,
            (i, i)
        )
        
        # Forward past current cell
        i = jax.lax.while_loop(
            lambda k: (k < t_len) & (t[k] == c),
            lambda k: k + 1,
            i
        )
        
        # 2. Skip covered cells
        i = jax.lax.while_loop(
            lambda k: (k < t_len) & cov[t[k]],
            lambda k: k + 1,
            i
        )
        
        # Forward past current cell again if needed
        i = jax.lax.while_loop(
            lambda k: (k < t_len) & (t[k] == c),
            lambda k: k + 1,
            i
        )
        
        return i

    # Vectorize advance_single
    v_advance = jax.vmap(jax.vmap(advance_single, in_axes=(0, 0, 0, 0, 0)), in_axes=(0, 0, 0, 0, 0))
    # Wait, cov is (E, C), so in_axes for cov is (0) for the outer vmap, and (None) for the inner!
    v_advance = jax.vmap(
        jax.vmap(advance_single, in_axes=(0, 0, 0, 0, None)), 
        in_axes=(0, 0, 0, 0, 0)
    )
    
    new_idx = v_advance(tours, tour_lens, idx, cell, covered)
    return new_idx


def jax_guide_update(state: JaxGuideState, positions, coverage_grid, graph_neighbors, n_cells, cell_size, revisit_penalty=0.5):
    """
    positions: (E, N, 2)
    coverage_grid: (E, H, W)
    """
    E, N = state.idx.shape
    C = n_cells
    
    # Calculate cell from pos
    # Wait, graph.cell_of does:
    # c = clip(pos[:, 0] / cell_size) ...
    # We'll assume the caller passes the cell indices directly to keep it clean.
    pass

def jax_guide_update(state: JaxGuideState, positions, coverage_grid, graph_neighbors, w, h, cell_size, revisit_penalty=0.5):
    E, N = state.idx.shape
    C = w * h
    
    # 1. cell_of
    col = jnp.clip((positions[..., 0] / cell_size).astype(jnp.int32), 0, w - 1)
    row = jnp.clip((positions[..., 1] / cell_size).astype(jnp.int32), 0, h - 1)
    cell = row * w + col # (E, N)
    
    covered = (coverage_grid > 0.5).reshape(E, C)
    
    reached = (state.target >= 0) & (cell == state.target)
    
    # 2. Advance index for all robots (even if they didn't change cell, to be safe, or we can condition it)
    new_idx = advance_cursors(state.tours, state.tour_lens, state.idx, cell, covered)
    
    # Extract new target cell from tour
    # If new_idx >= tour_lens, target is -1 (needs replan)
    # We will use pure_callback for replan later, but for now just output -1
    is_finished = new_idx >= state.tour_lens
    
    # Gather target from tours
    e_idx = jnp.arange(E)[:, None]
    n_idx = jnp.arange(N)[None, :]
    
    # Safe index
    safe_idx = jnp.minimum(new_idx, state.tour_lens - 1)
    tour_target = state.tours[e_idx, n_idx, safe_idx]
    
    tour_target = jnp.where(is_finished, -1, tour_target)
    
    # 3. Pathfinding (Bellman-Ford) from tour_target to all cells
    cost = jnp.where(covered, 1.0 + revisit_penalty, 1.0)
    
    # We run Bellman-Ford for targets >= 0
    dist = jax_bellman_ford_to_target(tour_target, cost, graph_neighbors, C)
    
    # 4. Find the next waypoint from current cell
    waypoint = get_next_waypoint(cell, dist, cost, graph_neighbors, C)
    
    # If finished or unreachable, keep -1
    waypoint = jnp.where(is_finished, -1, waypoint)
    
    new_state = state._replace(
        target=waypoint,
        prev_cell=cell,
        idx=new_idx
    )
    
    return new_state, waypoint, reached

import numpy as np

def reset_guides_callback(done, pos, current_tours, current_tour_lens, guides):
    """
    done: (E,) bool
    pos: (E, N, 2)
    guides: list of BoscoGuide objects
    """
    done = np.asarray(done)
    pos = np.asarray(pos)
    
    new_tours = np.copy(current_tours)
    new_tour_lens = np.copy(current_tour_lens)
    
    for e in range(len(done)):
        if done[e]:
            guides[e].reset(pos[e])
            
            for r in range(pos.shape[1]):
                tour = guides[e].tours[r]
                t_len = len(tour)
                new_tour_lens[e, r] = t_len
                # pad tour
                padded = np.full(current_tours.shape[2], -1, dtype=np.int32)
                padded[:t_len] = tour[:current_tours.shape[2]]
                new_tours[e, r] = padded
                
    return new_tours, new_tour_lens

def jax_reset_guides(done, pos, tours, tour_lens, guides):
    # Use jax.pure_callback to call the Python function
    # pure_callback requires shape_dtypes for the output
    out_shapes = (
        jax.ShapeDtypeStruct(tours.shape, tours.dtype),
        jax.ShapeDtypeStruct(tour_lens.shape, tour_lens.dtype)
    )
    
    # We pass 'guides' via closure, pure_callback doesn't allow non-JAX objects as args
    def callback(d, p, t, tl):
        return reset_guides_callback(d, p, t, tl, guides)
        
    return jax.pure_callback(callback, out_shapes, done, pos, tours, tour_lens)


def jax_guide_step(state: JaxGuideState, positions, coverage_grid, done, graph_neighbors, w, h, cell_size, guides, revisit_penalty=0.5):
    """
    1. Handle resets via pure_callback
    2. Advance cursors and compute target
    """
    E, N = state.idx.shape
    C = w * h
    
    def do_reset():
        return jax_reset_guides(done, positions, state.tours, state.tour_lens, guides)
        
    def skip_reset():
        return state.tours, state.tour_lens
        
    new_tours, new_tour_lens = jax.lax.cond(
        jnp.any(done),
        do_reset,
        skip_reset
    )
    
    # Also reset idx, target, prev_cell, fail_cov for done envs
    idx = jnp.where(done[:, None], 0, state.idx)
    target = jnp.where(done[:, None], -1, state.target)
    prev_cell = jnp.where(done[:, None], -1, state.prev_cell)
    fail_cov = jnp.where(done[:, None], -1, state.fail_cov)
    
    state = state._replace(
        tours=new_tours,
        tour_lens=new_tour_lens,
        idx=idx,
        target=target,
        prev_cell=prev_cell,
        fail_cov=fail_cov
    )
    
    # 2. Update logic
    col = jnp.clip((positions[..., 0] / cell_size).astype(jnp.int32), 0, w - 1)
    row = jnp.clip((positions[..., 1] / cell_size).astype(jnp.int32), 0, h - 1)
    cell = row * w + col
    
    covered = (coverage_grid > 0.5).reshape(E, C)
    
    reached = (state.target >= 0) & (cell == state.target)
    
    # Only advance if cell changed or stale (fail_cov != n_covered)
    # To keep JAX simple and avoid conditional branching across all, we just advance always.
    # It's fast anyway.
    new_idx = advance_cursors(state.tours, state.tour_lens, state.idx, cell, covered)
    
    is_finished = new_idx >= state.tour_lens
    
    e_idx = jnp.arange(E)[:, None]
    n_idx = jnp.arange(N)[None, :]
    safe_idx = jnp.minimum(new_idx, state.tour_lens - 1)
    tour_target = state.tours[e_idx, n_idx, safe_idx]
    tour_target = jnp.where(is_finished, -1, tour_target)
    
    cost = jnp.where(covered, 1.0 + revisit_penalty, 1.0)
    dist = jax_bellman_ford_to_target(tour_target, cost, graph_neighbors, C)
    
    waypoint = get_next_waypoint(cell, dist, cost, graph_neighbors, C)
    waypoint = jnp.where(is_finished, -1, waypoint)
    
    new_state = state._replace(
        target=waypoint,
        prev_cell=cell,
        idx=new_idx
    )
    
    return new_state, waypoint, reached

