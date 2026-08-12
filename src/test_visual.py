#!/usr/bin/env python3
"""Visualise a trained MAPPO policy in real time using pygame (JAX).

Usage (from the project root):
    python -m src.test_visual
    python -m src.test_visual --checkpoint checkpoints/checkpoint_final.pkl --episodes 10
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pygame

# Ensure the project root is on sys.path so that "from src.xxx import ..."
# works regardless of where the script is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp

from src.algorithms.mappo import RunningMeanStd, rms_normalize
from src.envs.coverage_vector_env import MultiRobotCoverageEnv
from src.models.actor_critic import Actor
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------
SCALE      = 80   # pixels per metre
MARGIN     = 40   # pixel border around the map
HUD_HEIGHT = 50   # pixels for the stats bar at the bottom
FPS        = 30

# One colour per robot (cycles if more than 4)
_ROBOT_COLORS = [
    (220,  60,  60),
    ( 60, 120, 220),
    ( 50, 180,  50),
    (220, 160,   0),
]

COLORS = {
    'bg':        (240, 240, 235),
    'covered':     (140, 220, 140),
    'uncovered':   (215, 215, 210),
    'unreachable': (170, 170, 165),
    'wall':      ( 55,  55,  55),
    'border':    ( 30,  30,  30),
    'grid':      (180, 180, 175),
    'lidar':     (255, 165,   0),
    'dead':      (120, 120, 120),
    'hud_bg':    ( 25,  25,  25),
    'hud_text':  (230, 230, 230),
}


def _to_px(x: float, y: float, map_h: float) -> tuple[int, int]:
    """World metres → pygame pixel (top-left origin, y-axis flipped)."""
    return (int(x * SCALE) + MARGIN,
            int((map_h - y) * SCALE) + MARGIN)


def _snapshot(env: MultiRobotCoverageEnv, state, want_lidar: bool) -> dict:
    """Pull the arrays needed for one frame back to the host in one go.

    Device→host transfers are the only per-frame cost of rendering, so every
    field is fetched with a single `jax.device_get` on a packed tuple.
    """
    info = env.get_info(state)
    payload = (
        state.robot_positions,
        state.robot_headings,
        state.coverage_grid,
        state.robot_alive,
        info['step'],
        info['coverage_ratio'],
        info['covered_cells'],
        info['total_cells'],
        env._cast_lidar_all(state.robot_positions, state.robot_headings)
        if want_lidar else jnp.zeros((0,), jnp.float32),
    )
    pos, hdg, grid, alive, step, cov_ratio, covered, total, lidar = jax.device_get(payload)
    return {
        'positions':      np.asarray(pos),
        'headings':       np.asarray(hdg),
        'coverage_grid':  np.asarray(grid),
        'alive':          np.asarray(alive),
        'step':           int(step),
        'coverage_ratio': float(cov_ratio),
        'covered_cells':  int(covered),
        'total_cells':    int(total),
        'lidar':          np.asarray(lidar),
    }


def _draw_frame(
    surface: pygame.Surface,
    env: MultiRobotCoverageEnv,
    walls: np.ndarray,
    snap: dict,
    font: pygame.font.Font,
    ep_reward: float,
    show_lidar: bool,
) -> None:
    mw = env.map_layout.width
    mh = env.map_layout.height
    cs = env.cell_size

    surface.fill(COLORS['bg'])

    # -- Coverage cells --
    # Unreachable cells are drawn apart from pending ones: they are excluded
    # from the coverage denominator, so leaving them "uncovered" would suggest
    # work that can never be done.
    cell_px = int(cs * SCALE)
    grid = snap['coverage_grid']
    free = env.free_mask_np
    for row in range(env.grid_h):
        for col in range(env.grid_w):
            if free[row, col] == 0.0:
                color = COLORS['unreachable']
            else:
                color = COLORS['covered'] if grid[row, col] > 0.0 else COLORS['uncovered']
            px, py = _to_px(col * cs, (row + 1) * cs, mh)
            pygame.draw.rect(surface, color, pygame.Rect(px, py, cell_px, cell_px))

    # -- Grid cell lines --
    map_px_w = int(mw * SCALE)
    map_px_h = int(mh * SCALE)
    for col in range(env.grid_w + 1):
        x = MARGIN + int(col * cell_px)
        pygame.draw.line(surface, COLORS['grid'], (x, MARGIN), (x, MARGIN + map_px_h))
    for row in range(env.grid_h + 1):
        y = MARGIN + int(row * cell_px)
        pygame.draw.line(surface, COLORS['grid'], (MARGIN, y), (MARGIN + map_px_w, y))

    # -- Walls --
    for x0, y0, x1, y1 in walls:
        px, py = _to_px(x0, y1, mh)
        w = int((x1 - x0) * SCALE)
        h = int((y1 - y0) * SCALE)
        pygame.draw.rect(surface, COLORS['wall'], pygame.Rect(px, py, w, h))

    # -- Map border --
    pygame.draw.rect(
        surface, COLORS['border'],
        pygame.Rect(MARGIN, MARGIN, map_px_w, map_px_h),
        2,
    )

    positions = snap['positions']
    headings  = snap['headings']

    # -- Lidar rays (toggle with 'L') --
    if show_lidar and snap['lidar'].size:
        angles_rel = np.linspace(0.0, 2.0 * np.pi, env.n_rays, endpoint=False)
        for i in range(env.num_robots):
            pos    = positions[i]
            angles = headings[i] + angles_rel
            dists  = snap['lidar'][i] * env.max_lidar_range   # stored normalised
            cx, cy = _to_px(pos[0], pos[1], mh)
            ex = pos[0] + dists * np.cos(angles)
            ey = pos[1] + dists * np.sin(angles)
            for x, y in zip(ex, ey):
                pygame.draw.line(surface, COLORS['lidar'], (cx, cy), _to_px(x, y, mh), 1)

    # -- Robots --
    r_px = max(5, int(env.robot_radius * SCALE))
    for i in range(env.num_robots):
        cx, cy = _to_px(positions[i, 0], positions[i, 1], mh)
        hdg    = headings[i]
        color  = (_ROBOT_COLORS[i % len(_ROBOT_COLORS)] if snap['alive'][i]
                  else COLORS['dead'])
        pygame.draw.circle(surface, color, (cx, cy), r_px)
        tip_x = cx + int(r_px * 1.8 * np.cos(hdg))
        tip_y = cy - int(r_px * 1.8 * np.sin(hdg))
        pygame.draw.line(surface, (255, 255, 255), (cx, cy), (tip_x, tip_y), 2)
        # robot index label
        lbl = font.render(str(i), True, (255, 255, 255))
        surface.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

    # -- HUD --
    text = (f"  Step {snap['step']:4d}/{env.max_steps}  |  "
            f"Coverage {snap['coverage_ratio']:.1%} "
            f"({snap['covered_cells']}/{snap['total_cells']})  |  "
            f"Ep Reward {ep_reward:8.2f}  |  "
            f"[ESC] quit  [R] reset  [L] lidar")
    win_h = surface.get_height()
    pygame.draw.rect(
        surface, COLORS['hud_bg'],
        pygame.Rect(0, win_h - HUD_HEIGHT, surface.get_width(), HUD_HEIGHT),
    )
    rendered = font.render(text, True, COLORS['hud_text'])
    surface.blit(rendered, (8, win_h - HUD_HEIGHT + (HUD_HEIGHT - rendered.get_height()) // 2))


def _build_policy_fn(env: MultiRobotCoverageEnv, actor: Actor):
    """One jitted call per frame: normalise obs → greedy action → env step.

    Fusing the policy and the environment transition keeps a single device
    round-trip per rendered frame; only the render snapshot comes back.
    """

    @jax.jit
    def policy_step(params, rms, state, obs):
        obs_n = rms_normalize(rms, obs) if rms is not None else obs
        mean, _ = actor.apply(params, obs_n)
        action = jnp.tanh(mean)                       # deterministic
        next_state, rewards, terminated, truncated = env.step(state, action)
        return next_state, env.get_obs(next_state), rewards, terminated, truncated

    return policy_step


def run_episode(
    env: MultiRobotCoverageEnv,
    policy_step,
    params,
    obs_rms: RunningMeanStd | None,
    key: jax.Array,
    surface: pygame.Surface,
    clock: pygame.time.Clock,
    font: pygame.font.Font,
    walls: np.ndarray,
    fps: int,
    view_state: dict,
) -> float | None:
    """Run one episode. Returns total reward, or None if the user quit."""
    state = env.reset(key)
    obs   = env.get_obs(state)

    ep_reward = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_r:
                    return ep_reward   # early reset
                if event.key == pygame.K_l:
                    view_state['show_lidar'] = not view_state['show_lidar']

        state, obs, rewards, terminated, truncated = policy_step(
            params, obs_rms, state, obs
        )
        ep_reward += float(jnp.mean(rewards))

        snap = _snapshot(env, state, view_state['show_lidar'])
        _draw_frame(surface, env, walls, snap, font, ep_reward, view_state['show_lidar'])
        pygame.display.flip()
        clock.tick(fps)

        if bool(terminated) or bool(truncated):
            return ep_reward


def _load_checkpoint(path: str, device: jax.Device) -> tuple[dict, RunningMeanStd | None, int]:
    """Read a JAX training checkpoint and place its arrays on `device`."""
    try:
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
    except (pickle.UnpicklingError, UnicodeDecodeError, EOFError) as exc:
        raise SystemExit(
            f"Cannot read '{path}' as a JAX checkpoint ({exc}).\n"
            "PyTorch-era '.pt' checkpoints are not loadable by the JAX policy: "
            "retrain with `python -m src.train_simple` to produce "
            "'checkpoints/checkpoint_final.pkl'."
        ) from exc

    if 'actor_params' not in ckpt:
        raise SystemExit(
            f"'{path}' has no 'actor_params' entry — it is not a JAX checkpoint "
            "written by src.train_simple."
        )

    params = jax.device_put(ckpt['actor_params'], device)
    rms = None
    if ckpt.get('obs_rms') is not None:
        rms = RunningMeanStd(*jax.device_put(tuple(ckpt['obs_rms']), device))
    return params, rms, int(ckpt.get('update', 0))


def main() -> None:
    _default_cfg  = os.path.join(_ROOT, 'config', 'mappo_baseline.yaml')
    _default_ckpt = os.path.join(_ROOT, 'checkpoints', 'checkpoint_final.pkl')

    parser = argparse.ArgumentParser(description='Visualise trained MAPPO policy with pygame')
    parser.add_argument('--checkpoint', default=_default_ckpt,
                        help='Path to .pkl checkpoint (default: checkpoints/checkpoint_final.pkl)')
    parser.add_argument('--config',     default=_default_cfg,
                        help='Path to YAML config file')
    parser.add_argument('--episodes',   type=int, default=0,
                        help='Episodes to run; 0 = loop forever (default: 0)')
    parser.add_argument('--fps',        type=int, default=FPS,
                        help=f'Rendering FPS (default: {FPS})')
    parser.add_argument('--seed',       type=int, default=0,
                        help='PRNG seed for episode resets (default: 0)')
    parser.add_argument('--no-obs-norm', action='store_true',
                        help='Disable observation normalisation')
    parser.add_argument('--backend',    default='cpu',
                        choices=['auto', 'metal', 'cuda', 'gpu', 'cpu'],
                        help='JAX backend. Default "cpu": a single-env rollout is '
                             'tiny, so the CPU beats accelerator launch overhead')
    args = parser.parse_args()

    config    = load_config(args.config)
    env_cfg   = config.get('env',   {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    # Must run before any array is created so implicit placement follows it.
    device = select_device(None if args.backend == 'auto' else args.backend)
    print(f"Device: {describe(device)}")

    env   = MultiRobotCoverageEnv(env_cfg)
    walls = np.asarray(env.walls)

    actor = Actor(
        action_dim=env.action_dim,
        vec_dim=env.obs_vec_dim,
        n_rays=env.n_rays,
        patch_dim=env.patch_dim,
        lidar_embed=model_cfg.get('lidar_embed',  64),
        hidden_size=model_cfg.get('hidden_size', 128),
    )

    params, obs_rms, update = _load_checkpoint(args.checkpoint, device)
    print(f"Loaded: {args.checkpoint}  (update {update})")

    if args.no_obs_norm or not train_cfg.get('normalize_obs', True):
        obs_rms = None
        print("Observation normalisation: disabled.")
    elif obs_rms is None:
        print("Warning: checkpoint has no obs_rms — running without normalisation.")
    else:
        print("Observation normalisation: loaded from checkpoint.")

    policy_step = _build_policy_fn(env, actor)

    # -- Pygame setup --
    mw    = env.map_layout.width
    mh    = env.map_layout.height
    win_w = int(mw * SCALE) + 2 * MARGIN
    win_h = int(mh * SCALE) + 2 * MARGIN + HUD_HEIGHT

    pygame.init()
    surface = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption('MAPPO Coverage — Visual Test')
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont('monospace', 15)

    view_state = {'show_lidar': False}
    key = jax.random.PRNGKey(args.seed)

    ep_num = 0
    try:
        while args.episodes == 0 or ep_num < args.episodes:
            key, ep_key = jax.random.split(key)
            print(f"Episode {ep_num + 1} ...", end='', flush=True)
            ret = run_episode(env, policy_step, params, obs_rms, ep_key,
                              surface, clock, font, walls, args.fps, view_state)
            if ret is None:
                print("  (quit)")
                break
            print(f"  total reward = {ret:.2f}")
            ep_num += 1
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
