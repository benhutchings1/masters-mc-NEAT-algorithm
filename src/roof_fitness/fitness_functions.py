import numpy as np
import neat
from typing import List
import warnings
import math
from src.logger import StructLogger

def structure_fitness(input:list, outputs:List[np.array], logger:StructLogger) \
    -> List[float]:
    # Setup logging file to first time with headers
    if logger.first_time:
        setup_logger(logger)
        logger.first_time = False    
    
    fit_funcs = [fit_complexity, fit_symmetry]
    fitnesses = []
    for output in outputs:
        g_fit = []
        # Check no values are above height or below 0
        if not fit_compliance(input, output):
            g_fit.append(0)
        else:
            g_fit.append(1.0)
        # Get other fitness values
        for f in fit_funcs:
            g_fit.append(f(input, output))
        fitnesses.append(np.average(g_fit))
        # Log individual values
        logger.log_value(g_fit)  
    return fitnesses

def setup_logger(logger:StructLogger):
    f_names = ["fit_compliance", "fit_complexity", "fit_symmetry"]
    logger.add_header(f_names)
    logger.start_gen()

def single_structure_fitness(input:list, output:np.array):
    fit_funcs = [fit_complexity, fit_symmetry]
    f_names = []
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
    count = 0
    comp_val = 0
    for yi, y in enumerate(output):
        for xi, x in enumerate(y):
            surr = get_surrounding(output, yi, xi)
            grad = [s - output[yi][xi] for s in surr]
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
    return np.array(out)
   
def fit_symmetry(input:np.array, output:np.array) -> float:
    return max(fit_vert_symmetry(output), fit_horiz_symmetry(output))

def fit_vert_symmetry(output:np.array) -> float:
    # Get sides for symmetry checking
    a = None
    b = None
    same = 0
    a, b = arr_split(output, axis=1)
    # Flip b side
    b = np.flip(b, axis=1)
    same += np.count_nonzero(a==b)

    return same/(a.shape[0] * a.shape[1])


def fit_horiz_symmetry(output:np.array) -> float:
    # Get sides for symmetry checking
    same = 0
    a, b = arr_split(output, axis=0)
    # Flip b side to directly compare
    b = np.flip(b, axis=0)
    same += np.count_nonzero(a==b)  
    return same/(a.shape[0] * a.shape[1])

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

def fit_compliance(input:np.array, output:np.array):   
    return not (np.min(output) < 0 or np.max(output) > input[0])
    