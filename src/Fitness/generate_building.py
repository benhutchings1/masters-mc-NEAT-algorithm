import numpy as np


def generate(model, height:int, length:int, width:int):
    """
    Uses a given model to create a house to the specified height, length, width
    Get the model to predict a block by giving the surrounding blocks
    """
    out = np.zeros((height, length, width)).astype(int)
    for h in range(height):
        for l in range(length):
            for w in range(width):
                surr_points = get_surrounding_points(out, w, h, l)
                # inputs = ?
                # Use model to predict current point
                # model.activate(inputs)

def get_surrounding_points(ar:np.array, x:int, y:int, z:int) -> np.array:
    out = np.zeros((2 * 3 * 3) - 1).astype(int)
    out.fill(-1)
    idx = 0
    for yi in range(0, -2, -1):
        # Check if y row exists
        if not y + yi < 0:
            for zi in range(-1, 2):
                # Check if z is within bounds of array
                if zi + z >= 0 and zi + z < ar.shape[1]:
                    for xi in range(-1, 2):
                        if xi + x >= 0 and xi + x < ar.shape[2]:
                            if xi == 0 and zi == 0 and yi == 0:
                                pass
                            else:
                                out[idx] = ar[y+yi][z+zi][x+xi]
                                idx += 1                            
                        else:
                            idx += 1
                else:
                    idx += 3   
                
    return out