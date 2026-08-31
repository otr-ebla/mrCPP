def get_valid_door_cells(cells, door_cells):
    valid = []
    for dc in range(cells - door_cells + 1):
        w1 = dc
        w2 = cells - dc - door_cells
        if (w1 == 0 or w1 >= 2) and (w2 == 0 or w2 >= 2):
            valid.append(dc)
    return valid

print(f"door_cells = 3")
for cells in range(3, 10):
    print(f"cells = {cells}: {get_valid_door_cells(cells, 3)}")
