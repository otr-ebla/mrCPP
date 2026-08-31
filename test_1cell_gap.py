import numpy as np
from src.envs.map_layouts import ProceduralMapLayout
import random

def get_cells(w_seg):
    return (
        round(w_seg[0]/0.5), round(w_seg[1]/0.5),
        round(w_seg[2]/0.5), round(w_seg[3]/0.5)
    )

def check_gaps():
    for seed in range(100):
        random.seed(seed)
        np.random.seed(seed)
        layout = ProceduralMapLayout()
        
        for w1 in layout.walls:
            if w1[0] < -1: continue
            c1 = get_cells(w1)
            is_w1_vert = abs(c1[0] - c1[2]) == 0
            
            for w2 in layout.walls:
                if w1 == w2: continue
                if w2[0] < -1: continue
                c2 = get_cells(w2)
                is_w2_vert = abs(c2[0] - c2[2]) == 0
                
                # If w1 is vertical and w2 is horizontal
                if is_w1_vert and not is_w2_vert:
                    x_v = c1[0]
                    y_h = c2[1]
                    
                    # They intersect if y_h is between c1[1] and c1[3], and x_v is between c2[0] and c2[2]
                    # The gap happens if one endpoint is just 1 cell away from intersection.
                    # Actually, if y_h is 1 cell away from c1[1] or c1[3] AND x_v is between c2[0] and c2[2]:
                    if c2[0] <= x_v <= c2[2]:
                        if abs(y_h - c1[1]) == 1 or abs(y_h - c1[3]) == 1:
                            print(f"Seed {seed}: 1-cell gap between vertical {c1} and horizontal {c2}")
                            return True
                    if c1[1] <= y_h <= c1[3]:
                        if abs(x_v - c2[0]) == 1 or abs(x_v - c2[2]) == 1:
                            print(f"Seed {seed}: 1-cell gap between vertical {c1} and horizontal {c2}")
                            return True
    return False

if check_gaps():
    print("FAILED")
else:
    print("PASSED")
