import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ProceduralMapLayout:
    """
    Generates a random indoor map layout using Binary Space Partitioning (BSP).
    Ensures a varied number of rooms, corridors, and doorways while keeping
    computational complexity low. 
    """
    def __init__(self, width=12.0, height=8.0, min_room_size=3.0, max_walls=30,
                 cell_size: float = 0.5, robot_radius: float = 0.20):
        self.width = width
        self.height = height
        self.min_room_size = min_room_size
        self.max_walls = max_walls
        
        self.outer_t = 0.20
        self.inner_t = 0.08
        # Guarantee at least 2 coverable cells across every doorway,
        # regardless of grid alignment.
        self.door_size = 2 * cell_size + 2 * robot_radius
        
        self.walls = []
        self._generate_layout()
        self._pad_walls()

    def _generate_layout(self):
        # 1. Outer boundaries
        self.walls.append([-self.outer_t, -self.outer_t, self.width + self.outer_t, 0.0])
        self.walls.append([-self.outer_t, self.height, self.width + self.outer_t, self.height + self.outer_t])
        self.walls.append([-self.outer_t, -self.outer_t, 0.0, self.height + self.outer_t])
        self.walls.append([self.width, -self.outer_t, self.width + self.outer_t, self.height + self.outer_t])
        
        # 2. Recursive BSP to create rooms
        self._split_space(0.0, 0.0, self.width, self.height, depth=0, max_depth=3)

    def _split_space(self, x, y, w, h, depth, max_depth):
        if depth >= max_depth:
            return

        # Determine split direction dynamically based on aspect ratio
        split_horizontally = w < h
        if w >= h * 1.2:
            split_horizontally = False
        elif h >= w * 1.2:
            split_horizontally = True
        else:
            split_horizontally = random.choice([True, False])

        if split_horizontally:
            if h < self.min_room_size * 2:
                return
            split_y = y + random.uniform(self.min_room_size, h - self.min_room_size)
            
            # Wall segments with a door gap
            door_pos = x + random.uniform(0.5, w - self.door_size - 0.5)
            self.walls.append([x, split_y - self.inner_t / 2, door_pos, split_y + self.inner_t / 2])
            self.walls.append([door_pos + self.door_size, split_y - self.inner_t / 2, x + w, split_y + self.inner_t / 2])
            
            # Recurse into the two new sub-regions
            self._split_space(x, y, w, split_y - y, depth + 1, max_depth)
            self._split_space(x, split_y, w, y + h - split_y, depth + 1, max_depth)
            
        else:
            if w < self.min_room_size * 2:
                return
            split_x = x + random.uniform(self.min_room_size, w - self.min_room_size)
            
            # Wall segments with a door gap
            door_pos = y + random.uniform(0.5, h - self.door_size - 0.5)
            self.walls.append([split_x - self.inner_t / 2, y, split_x + self.inner_t / 2, door_pos])
            self.walls.append([split_x - self.inner_t / 2, door_pos + self.door_size, split_x + self.inner_t / 2, y + h])
            
            # Recurse into the two new sub-regions
            self._split_space(x, y, split_x - x, h, depth + 1, max_depth)
            self._split_space(split_x, y, w - (split_x - x), h, depth + 1, max_depth)

    def _pad_walls(self):
        """
        Pads the walls array to a static maximum size for JAX compatibility.
        Missing walls are placed at coordinates that will never interfere with the scene.
        """
        current_walls = len(self.walls)
        if current_walls > self.max_walls:
            self.walls = self.walls[:self.max_walls]
        else:
            # Pad with dummy zero-area walls far outside the map
            padding = [[-10.0, -10.0, -10.0, -10.0] for _ in range(self.max_walls - current_walls)]
            self.walls.extend(padding)

    def get_walls(self):
        return np.array(self.walls, dtype=np.float32)


def create_map_bank(num_maps=16, **kwargs):
    """
    Generates a bank of random maps to be cached by the JAX environment.
    Runs purely in Python during initialization to avoid JAX dynamic loop overhead.
    """
    return [ProceduralMapLayout(**kwargs) for _ in range(num_maps)]


def main():
    # Test generation and visualization
    layout = ProceduralMapLayout()
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
        # Skip padding dummy walls
        if x1 <= -5.0:
            continue
            
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
    ax.set_title("Procedural Indoor Map Layout")
    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

    plt.tight_layout()
    plt.savefig("procedural_room_layout.png", dpi=150)
    print("Saved procedural_room_layout.png")


if __name__ == "__main__":
    main()