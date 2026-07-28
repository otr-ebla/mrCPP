#!/usr/bin/env python3
"""Visualise a trained MAPPO policy in real time using pygame.

Usage (from internal-simple-project/):
    python src/test_visual.py
    python src/test_visual.py --checkpoint checkpoints/best_policy.pt --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import pygame

# Ensure the project root (internal-simple-project/) is on sys.path so that
# "from src.xxx import ..." works regardless of where the script is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.envs.coverage_vector_env import MultiRobotCoverageEnv
from src.models.actor_critic_mlp import Actor
from src.utils.config_parser import load_config
from src.train_simple import RunningMeanStd

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------
SCALE      = 80   # pixels per metre
MARGIN     = 40   # pixel border around the map
HUD_HEIGHT = 50   # pixels for the stats bar at the bottom
FPS        = 30

# One colour per robot (cycles if more than 3)
_ROBOT_COLORS = [
    (220,  60,  60),
    ( 60, 120, 220),
    ( 50, 180,  50),
    (220, 160,   0),
]

COLORS = {
    'bg':        (240, 240, 235),
    'covered':   (140, 220, 140),
    'uncovered': (215, 215, 210),
    'wall':      ( 55,  55,  55),
    'border':    ( 30,  30,  30),
    'grid':      (180, 180, 175),
    'lidar':     (255, 165,   0),
    'hud_bg':    ( 25,  25,  25),
    'hud_text':  (230, 230, 230),
}


def _to_px(x: float, y: float, map_h: float) -> tuple[int, int]:
    """World metres → pygame pixel (top-left origin, y-axis flipped)."""
    return (int(x * SCALE) + MARGIN,
            int((map_h - y) * SCALE) + MARGIN)


def _draw_frame(
    surface: pygame.Surface,
    env: MultiRobotCoverageEnv,
    font: pygame.font.Font,
    ep_reward: float,
    show_lidar: bool,
) -> None:
    mw = env.map_layout.width
    mh = env.map_layout.height
    cs = env.cell_size

    surface.fill(COLORS['bg'])

    # -- Coverage cells --
    cell_px = int(cs * SCALE)
    for row in range(env.grid_h):
        for col in range(env.grid_w):
            color = COLORS['covered'] if env.coverage_grid[row, col] else COLORS['uncovered']
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
    for x0, y0, x1, y1 in env.walls:
        px, py = _to_px(x0, y1, mh)
        w = int((x1 - x0) * SCALE)
        h = int((y1 - y0) * SCALE)
        pygame.draw.rect(surface, COLORS['wall'], pygame.Rect(px, py, w, h))

    # -- Map border --
    pygame.draw.rect(
        surface, COLORS['border'],
        pygame.Rect(MARGIN, MARGIN, int(mw * SCALE), int(mh * SCALE)),
        2,
    )

    # -- Lidar rays (toggle with 'L') --
    if show_lidar:
        for i in range(env.num_robots):
            pos       = env.robot_positions[i]
            hdg       = env.robot_headings[i]
            other_pos = np.delete(env.robot_positions, i, axis=0)
            ranges    = env._cast_lidar(pos, hdg, other_pos)   # normalised [0, 1]
            angles    = hdg + np.linspace(0.0, 2.0 * np.pi, env.n_rays, endpoint=False)
            cx, cy    = _to_px(pos[0], pos[1], mh)
            for ang, r in zip(angles, ranges):
                dist = r * env.max_lidar_range
                ex   = pos[0] + dist * np.cos(ang)
                ey   = pos[1] + dist * np.sin(ang)
                ex_px, ey_px = _to_px(ex, ey, mh)
                pygame.draw.line(surface, COLORS['lidar'], (cx, cy), (ex_px, ey_px), 1)

    # -- Robots --
    r_px = max(5, int(env.robot_radius * SCALE))
    for i in range(env.num_robots):
        cx, cy = _to_px(env.robot_positions[i, 0], env.robot_positions[i, 1], mh)
        hdg     = env.robot_headings[i]
        color   = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
        pygame.draw.circle(surface, color, (cx, cy), r_px)
        tip_x = cx + int(r_px * 1.8 * np.cos(hdg))
        tip_y = cy - int(r_px * 1.8 * np.sin(hdg))
        pygame.draw.line(surface, (255, 255, 255), (cx, cy), (tip_x, tip_y), 2)
        # robot index label
        lbl = font.render(str(i), True, (255, 255, 255))
        surface.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

    # -- HUD --
    info = env._get_info()
    text = (f"  Step {info['step']:4d}/{env.max_steps}  |  "
            f"Coverage {info['coverage_ratio']:.1%} "
            f"({info['covered_cells']}/{info['total_cells']})  |  "
            f"Ep Reward {ep_reward:8.2f}  |  "
            f"[ESC] quit  [R] reset  [L] lidar")
    win_h = surface.get_height()
    pygame.draw.rect(
        surface, COLORS['hud_bg'],
        pygame.Rect(0, win_h - HUD_HEIGHT, surface.get_width(), HUD_HEIGHT),
    )
    rendered = font.render(text, True, COLORS['hud_text'])
    surface.blit(rendered, (8, win_h - HUD_HEIGHT + (HUD_HEIGHT - rendered.get_height()) // 2))


@torch.no_grad()
def run_episode(
    env: MultiRobotCoverageEnv,
    actor: Actor,
    obs_rms: RunningMeanStd | None,
    device: torch.device,
    surface: pygame.Surface,
    clock: pygame.time.Clock,
    font: pygame.font.Font,
    fps: int,
    view_state: dict,
) -> float | None:
    """Run one episode. Returns total reward, or None if user quit."""
    obs, _ = env.reset()
    N  = env.num_robots
    hx = actor.init_hidden(N, device=str(device))

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

        obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
        obs_t  = torch.from_numpy(obs_in).float().to(device)   # (N, obs_dim)
        mean, _, hx = actor(obs_t, hx)
        actions = torch.tanh(mean).cpu().numpy()               # deterministic

        obs, rewards, terminated, truncated, _ = env.step(actions)
        ep_reward += float(rewards.mean())

        _draw_frame(surface, env, font, ep_reward, view_state['show_lidar'])
        pygame.display.flip()
        clock.tick(fps)

        if terminated or truncated:
            return ep_reward


def main() -> None:
    _default_cfg  = os.path.join(_ROOT, 'config', 'mappo_baseline.yaml')
    _default_ckpt = os.path.join(_ROOT, 'checkpoints', 'best_policy.pt')

    parser = argparse.ArgumentParser(description='Visualise trained MAPPO policy with pygame')
    parser.add_argument('--checkpoint', default=_default_ckpt,
                        help='Path to .pt checkpoint (default: checkpoints/best_policy.pt)')
    parser.add_argument('--config',     default=_default_cfg,
                        help='Path to YAML config file')
    parser.add_argument('--episodes',   type=int, default=0,
                        help='Episodes to run; 0 = loop forever (default: 0)')
    parser.add_argument('--fps',        type=int, default=FPS,
                        help=f'Rendering FPS (default: {FPS})')
    parser.add_argument('--no-obs-norm', action='store_true',
                        help='Disable observation normalisation')
    args = parser.parse_args()

    config    = load_config(args.config)
    env_cfg   = config.get('env',   {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env        = MultiRobotCoverageEnv(env_cfg)
    obs_dim    = env.obs_dim
    action_dim = 2
    N          = env.num_robots

    actor = Actor(
        obs_dim, action_dim,
        model_cfg.get('hidden_size', 128),
        model_cfg.get('gru_hidden',  128),
    ).to(device)
    actor.eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt['actor_state'])
    print(f"Loaded: {args.checkpoint}  (update {ckpt.get('update', '?')})")

    obs_rms = None
    if not args.no_obs_norm and train_cfg.get('normalize_obs', True):
        if 'obs_rms' in ckpt:
            obs_rms = RunningMeanStd(shape=(obs_dim,))
            obs_rms.load_state_dict(ckpt['obs_rms'])
            print("Observation normalisation: loaded from checkpoint.")
        else:
            print("Warning: checkpoint has no obs_rms — running without normalisation.")

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

    ep_num = 0
    try:
        while args.episodes == 0 or ep_num < args.episodes:
            print(f"Episode {ep_num + 1} ...", end='', flush=True)
            ret = run_episode(env, actor, obs_rms, device, surface, clock, font, args.fps, view_state)
            if ret is None:
                print("  (quit)")
                break
            print(f"  total reward = {ret:.2f}")
            ep_num += 1
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
