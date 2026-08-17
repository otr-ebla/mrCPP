import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class IndoorMapLayout:
    def __init__(self):
        # Outer boundary dimensions in meters
        self.width = 12.0
        self.height = 8.0

        # Wall thicknesses.
        # Inner walls run along cell boundaries (x=4.0, x=6.0, y=4.0), so the
        # nearest cell centres sit 0.5*cell_size = 0.25 m away. A cell counts as
        # coverable only if the robot body (radius 0.20 m) fits at its centre, so
        # the inner half-thickness must stay below 0.25 - 0.20 = 0.05 m; anything
        # thicker sterilises the two cell lines flanking every inner wall. 0.08 m
        # total keeps a 0.01 m margin against floating-point ties.
        # The outer walls sit entirely outside [0, width] x [0, height], so their
        # thickness never eats into the play area and is left as-is.
        outer_t = 0.20
        inner_t = 0.08
        self.outer_t = outer_t
        self.inner_t = inner_t

        self.walls = []

        # ---------------------------------------------------------
        # 1. OUTER BOUNDARY WALLS
        # ---------------------------------------------------------
        # Pushed fully outside the [0, width] x [0, height] play area so
        # they only touch the border of the outermost grid cells (flush
        # with x=0 / x=width / y=0 / y=height) instead of overlapping
        # them. This keeps border cells entirely free to traverse/cover,
        # rather than partially occluded by wall thickness.
        self.walls.append([-outer_t, -outer_t, self.width + outer_t, 0.0])          # Bottom Wall
        self.walls.append([-outer_t, self.height, self.width + outer_t, self.height + outer_t])  # Top Wall
        self.walls.append([-outer_t, -outer_t, 0.0, self.height + outer_t])          # Left Wall
        self.walls.append([self.width, -outer_t, self.width + outer_t, self.height + outer_t])  # Right Wall

        # ---------------------------------------------------------
        # 2. INNER WALL: LEFT ROOM SEPARATOR (Vertical split at x=4.0)
        # ---------------------------------------------------------
        x_left_wall = 4.0
        x_l = x_left_wall - (inner_t / 2)
        x_r = x_left_wall + (inner_t / 2)

        y_mid_wall = 4.0
        y_b = y_mid_wall - (inner_t / 2)
        y_t = y_mid_wall + (inner_t / 2)

        self.walls.append([x_l, 0.2, x_r, 1.2])
        self.walls.append([x_l, 2.2, x_r, y_b])
        self.walls.append([x_l, y_t, x_r, 5.0])
        self.walls.append([x_l, 6.0, x_r, 7.8])

        # ---------------------------------------------------------
        # 3. INNER WALL: HORIZONTAL SPLIT FOR LEFT ROOMS (At y=4.0)
        # ---------------------------------------------------------
        self.walls.append([0.2, y_b, x_l, y_t])

        # ---------------------------------------------------------
        # 4. INNER WALL: RIGHT HALLWAY SEPARATOR (Vertical split at x=6.0)
        # ---------------------------------------------------------
        x_right_wall = 6.0
        x_rl = x_right_wall - (inner_t / 2)
        x_rr = x_right_wall + (inner_t / 2)

        self.walls.append([x_rl, 0.2, x_rr, 1.5])
        self.walls.append([x_rl, 2.5, x_rr, 5.5])
        self.walls.append([x_rl, 6.5, x_rr, 7.8])

        # ---------------------------------------------------------
        # 5. INNER WALL: HORIZONTAL DIVIDER WITH DOOR IN BIG RIGHT ROOM (At y=4.0)
        # ---------------------------------------------------------
        # Two segments leave a 1.0 m gap (door opening) between x=8.5 and x=9.5
        self.walls.append([x_rr, y_b, 8.5, y_t])
        self.walls.append([9.5, y_b, 11.8, y_t])

    def get_walls(self):
        return np.array(self.walls)


def main():
    layout = IndoorMapLayout()
    walls = layout.get_walls()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Light floor background
    floor = patches.Rectangle(
        (0, 0), layout.width, layout.height,
        facecolor="#f5f5f0", edgecolor="none", zorder=0
    )
    ax.add_patch(floor)

    # Draw each wall as a filled rectangle
    for x1, y1, x2, y2 in walls:
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            facecolor="#444444", edgecolor="black", linewidth=0.5, zorder=2
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.5, layout.width + 0.5)
    ax.set_ylim(-0.5, layout.height + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Indoor Map Layout")
    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/room_layout.png", dpi=150)
    print("Saved room_layout.png")


if __name__ == "__main__":
    main()