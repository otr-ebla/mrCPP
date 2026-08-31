import numpy as np

cell_size = 0.5
walls = [
    [4.0 - 0.04, 0.0, 4.0 + 0.04, 2.0],
    [4.0 - 0.04, 3.5, 4.0 + 0.04, 8.0]
] # vertical wall at x=4.0, door from 2.0 to 3.5

x = 4.0
w = 6.0
# We want to split horizontally at some split_y
y_endpoints = []
for w_seg in walls:
    # check if vertical
    if abs(w_seg[2] - w_seg[0]) < 0.2: # inner_t is 0.08, outer_t is 0.20
        # check if on boundary x or x+w
        cx = (w_seg[0] + w_seg[2]) / 2.0
        if abs(cx - x) < 0.3 or abs(cx - (x + w)) < 0.3:
            y_endpoints.append(round(w_seg[1] / cell_size))
            y_endpoints.append(round(w_seg[3] / cell_size))

print("y_endpoints (in cells):", y_endpoints)

for split_cell_y in range(1, 15):
    valid = True
    for y_ep in y_endpoints:
        if abs(split_cell_y - y_ep) == 1:
            valid = False
            break
    print(f"split_cell_y = {split_cell_y} (y = {split_cell_y * cell_size}): {'VALID' if valid else 'INVALID'}")
