import numpy as np
import neat
from typing import List
import warnings
import math

# Covers building
# Sloped sides

def structure_fitness(genomes:List[neat.DefaultGenome], input:list, outputs:List[np.array]) \
    -> List[float]:
    fit_funcs = [fit_cover, fit_complexity, fit_symmetry]
    fitnesses = []
    for output in outputs:
        g_fit = []
        for f in fit_funcs:
            g_fit.append(f(input, output))
        fitnesses.append(np.average(g_fit))
    return fitnesses

def single_structure_fitness(input:list, output:np.array):
    fit_funcs = [fit_cover, fit_complexity, fit_symmetry]
    f_names = ["Cover Fitness", "Complexity Fitness", "Symmetry Fitness"]
    fitness = {}
    for name, f in zip(f_names, fit_funcs):
        fitness[name] = f(None, input, output)
    return fitness

def get_roof_heighmap(output:np.array):
    assert len(output.shape) == 3
    height_map = np.zeros((output.shape[1], output.shape[2])).astype(int)
    for zi in range(output.shape[1]):
        for xi in range(output.shape[2]):
            # Iterate downwards to get top point
            for yi in range(output.shape[0]-1, -1, -1):
                if output[yi][zi][xi] != 0:
                    height_map[zi][xi] = yi + 1
                    break
    return height_map


def fit_complexity(input, output):
    heightmap = get_roof_heighmap(output)
    count = 0
    comp_val = 0
    for yi, y in enumerate(heightmap):
        for xi, x in enumerate(y):
            surr = get_surrounding(heightmap, yi, xi)
            grad = [s - heightmap[yi][xi] for s in surr]
            for val in grad:
                if val == -1 or val == 1:
                    comp_val += 1
                count += 1
    return comp_val / count
        
def get_surrounding(heightmap, yi, xi):
    out = []
    for i in range(3):
        for j in range(3):
            y = yi-1+i
            x = xi-1+j
            if not y < 0 and not y > len(heightmap) - 1:
                if not x < 0 and not x > len(heightmap[0]) - 1:
                    out.append(heightmap[y][x])
    return out
      


def fit_cover(input:np.array, output:np.array):
    # Get columns covered by a block
    return np.count_nonzero(get_roof_heighmap(output)) / (output.shape[1] * output.shape[2]) 
    


def fit_symmetry(input:np.array, output:np.array) -> float:
    return max(fit_vert_symmetry(output), fit_horiz_symmetry(output))

def fit_vert_symmetry(output:np.array) -> float:
    # Get sides for symmetry checking
    a = None
    b = None
    same = 0
    for y in output:
        a, b = arr_split(y, axis=1)
        # Flip b side
        b = np.flip(b, axis=1)
        same += np.count_nonzero(a==b)

    return same/(a.shape[0] * a.shape[1] * len(output))


def fit_horiz_symmetry(output:np.array) -> float:
    # Get sides for symmetry checking
    same = 0
    for y in output:
        a, b = arr_split(y, axis=0)
        # Flip b side
        b = np.flip(b, axis=0)
        same += np.count_nonzero(a==b)

    return same/(a.shape[0] * a.shape[1] * len(output))

def arr_split(plane:np.array, axis:int) -> List[np.array]:
    assert axis == 1 or axis == 0
    assert len(plane.shape) == 2
    
    left, right = None, None
    if axis == 0:
        # axis 0 split
        if plane.shape[0] % 2 == 0:
            div_line = int(plane.shape[0]/2)
            left = plane[:div_line]
            right = plane[div_line:]
            
        else:
            div_line = int((plane.shape[0]+1) / 2)
            left = plane[:div_line-1]
            right = plane[div_line:]
    else:
        if plane.shape[1] % 2 == 0:
            div_line = int(plane.shape[1]/2)
            left = plane[:, :div_line]
            right = plane[:, div_line:]
            
        else:
            div_line = int((plane.shape[1]+1) / 2)
            left = plane[:, :div_line-1]
            right = plane[:, div_line:]
    
    return left, right