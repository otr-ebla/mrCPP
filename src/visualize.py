"""
Pygame visualizer — random-policy robots with LiDAR ray overlay.

Controls
--------
R  : reset episode
Q  : quit
+  : speed up (fewer physics steps skipped)
-  : slow down
"""

import sys
import os
import math
import numpy as np
import pygame

# Allow running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from src.envs.coverage_vector_env import MultiRobotCoverageEnv

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------
PPM      = 80          # pixels per metre
MAP_W_M  = 12.0
MAP_H_M  = 8.0
WIN_W    = int(MAP_W_M * PPM)
WIN_H    = int(MAP_H_M * PPM)
FPS      = 30

# Palette
BG_COLOR     = (245, 245, 240)
GRID_COLOR   = (210, 210, 210)
WALL_COLOR   = (60,  60,  60)
COVERED_RGBA = (100, 210, 120, 70)    # semi-transparent green

ROBOT_COLORS = [
    (  70, 130, 180),   # steel blue
    ( 220,  70,  60),   # tomato red
    (  60, 180, 100),   # emerald green
    ( 180,  80, 200),   # purple
    ( 220, 160,  30),   # amber
]

def _lidar_rgba(robot_idx: int) -> tuple:
    r, g, b = ROBOT_COLORS[robot_idx % len(ROBOT_COLORS)]
    return (r, g, b, 80)        # same hue as robot, semi-transparent

def _hit_rgba(robot_idx: int) -> tuple:
    r, g, b = ROBOT_COLORS[robot_idx % len(ROBOT_COLORS)]
    return (r, g, b, 210)       # same hue, nearly opaque for hit dot


# ---------------------------------------------------------------------------
# Coordinate helpers  (world → screen, Y-axis flipped)
# ---------------------------------------------------------------------------

def w2s(x: float, y: float) -> tuple[int, int]:
    return int(x * PPM), int((MAP_H_M - y) * PPM)


def rect_w2s(x0: float, y0: float, x1: float, y1: float) -> pygame.Rect:
    """Convert world AABB [x0,y0,x1,y1] to screen pygame.Rect."""
    sx = int(x0 * PPM)
    sy = int((MAP_H_M - y1) * PPM)      # y1 is the higher world-y → smaller screen-y
    sw = max(1, int((x1 - x0) * PPM))
    sh = max(1, int((y1 - y0) * PPM))
    return pygame.Rect(sx, sy, sw, sh)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_grid(screen: pygame.Surface, env: MultiRobotCoverageEnv) -> None:
    cell_px = int(env.cell_size * PPM)
    for col in range(env.grid_w + 1):
        x = col * cell_px
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, WIN_H))
    for row in range(env.grid_h + 1):
        y = row * cell_px
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIN_W, y))


def draw_walls(screen: pygame.Surface, env: MultiRobotCoverageEnv) -> None:
    for wall in env.walls:
        pygame.draw.rect(screen, WALL_COLOR, rect_w2s(*wall))


def draw_coverage(cover_surf: pygame.Surface, env: MultiRobotCoverageEnv) -> None:
    cover_surf.fill((0, 0, 0, 0))
    cell_px = int(env.cell_size * PPM)
    for row in range(env.grid_h):
        for col in range(env.grid_w):
            if env.coverage_grid[row, col]:
                sx  = col * cell_px
                # row=0 → bottom of world → near bottom of screen
                sy  = (env.grid_h - row - 1) * cell_px
                pygame.draw.rect(cover_surf, COVERED_RGBA,
                                 pygame.Rect(sx, sy, cell_px, cell_px))


def draw_lidar(lidar_surf: pygame.Surface, env: MultiRobotCoverageEnv) -> None:
    lidar_surf.fill((0, 0, 0, 0))
    for i in range(env.num_robots):
        pos     = env.robot_positions[i]
        heading = env.robot_headings[i]
        angles  = heading + np.linspace(0.0, 2.0 * np.pi, env.n_rays, endpoint=False)
        dists_n = env._cast_lidar(pos, heading)            # normalised [0,1]
        dists   = dists_n * env.max_lidar_range

        ray_color = _lidar_rgba(i)
        hit_color = _hit_rgba(i)
        sx0, sy0  = w2s(pos[0], pos[1])
        for k in range(env.n_rays):
            hx = pos[0] + dists[k] * math.cos(angles[k])
            hy = pos[1] + dists[k] * math.sin(angles[k])
            sx1, sy1 = w2s(hx, hy)
            pygame.draw.line(lidar_surf, ray_color, (sx0, sy0), (sx1, sy1), 1)
            # Bright dot where ray hits a wall (not at max range)
            if dists_n[k] < 0.999:
                pygame.draw.circle(lidar_surf, hit_color, (sx1, sy1), 2)


def draw_robots(screen: pygame.Surface, env: MultiRobotCoverageEnv) -> None:
    r_px  = max(4, int(env.robot_radius * PPM))
    arr_l = int(r_px * 1.8)                        # heading arrow length

    for i in range(env.num_robots):
        pos     = env.robot_positions[i]
        heading = env.robot_headings[i]
        sx, sy  = w2s(pos[0], pos[1])
        color   = ROBOT_COLORS[i % len(ROBOT_COLORS)]

        # Filled circle
        pygame.draw.circle(screen, color, (sx, sy), r_px)
        # White border
        pygame.draw.circle(screen, (255, 255, 255), (sx, sy), r_px, 2)
        # Heading arrow
        ex = sx + int(arr_l * math.cos(heading))
        ey = sy - int(arr_l * math.sin(heading))   # screen Y is flipped
        pygame.draw.line(screen, (255, 255, 255), (sx, sy), (ex, ey), 2)
        # Arrow-head tip dot
        pygame.draw.circle(screen, (255, 255, 255), (ex, ey), 3)

        # Robot index label
        font_small = pygame.font.SysFont('monospace', 12, bold=True)
        lbl = font_small.render(str(i), True, (255, 255, 255))
        screen.blit(lbl, (sx - lbl.get_width() // 2, sy - lbl.get_height() // 2))


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_hud(screen: pygame.Surface, font: pygame.font.Font,
             info: dict, step: int, fps: float, speed: int) -> None:
    lines = [
        f"step {step:5d} / {info['total_cells'] and info.get('step', step)}",
        f"coverage  {info['coverage_ratio']:.1%}  "
        f"({info['covered_cells']}/{info['total_cells']} cells)",
        f"FPS {fps:.0f}   speed ×{speed}   [R]eset  [+/-] speed  [Q]uit",
    ]
    y = 8
    for line in lines:
        surf = font.render(line, True, (30, 30, 30))
        # White shadow for readability over light background
        shadow = font.render(line, True, (255, 255, 255))
        screen.blit(shadow, (9, y + 1))
        screen.blit(surf,   (8, y))
        y += font.get_height() + 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Multi-Robot Coverage — random policy + LiDAR")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont('monospace', 14)

    # Persistent transparent surfaces for alpha blending
    cover_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    lidar_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    env = MultiRobotCoverageEnv()
    env.reset(seed=42)

    step  = 0
    speed = 1      # physics steps per rendered frame

    running = True
    while running:
        # ---- Events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_r:
                    env.reset()
                    step = 0
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    speed = min(speed + 1, 10)
                if event.key == pygame.K_MINUS:
                    speed = max(speed - 1, 1)

        # ---- Physics (multiple steps per frame for speed) ----
        info = {}
        for _ in range(speed):
            actions = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(actions)
            step += 1
            if terminated or truncated:
                env.reset()
                step = 0

        # ---- Render ----
        screen.fill(BG_COLOR)
        draw_grid(screen, env)
        draw_coverage(cover_surf, env)
        screen.blit(cover_surf, (0, 0))
        draw_walls(screen, env)
        draw_lidar(lidar_surf, env)
        screen.blit(lidar_surf, (0, 0))
        draw_robots(screen, env)
        draw_hud(screen, font, info, step, clock.get_fps(), speed)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    main()
