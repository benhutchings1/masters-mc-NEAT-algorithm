import numpy as np
import neat
from typing import List
import warnings

# Check for windows
# Check for slanted roof? or flat roof?
# Size of model?
# House symmetry
# Fitness 0 = low, 1 = high

def structure_fitness(genomes:List[neat.DefaultGenome], input:list, outputs:List[np.array]) \
    -> List[float]:
    fit_funcs = [fit_seed_blocks, fit_airspace, fit_bounding_wall, fit_door, fit_symmetry]
    fitnesses = []
    for (__, genome), output in zip(genomes, outputs):
        g_fit = []
        for f in fit_funcs:
            g_fit.append(f(genome, input, output))
        fitnesses.append(np.average(g_fit))
    return fitnesses

def single_structure_fitness(input:list, output:np.array):
    fit_funcs = [fit_seed_blocks, fit_airspace, fit_bounding_wall, fit_door, fit_symmetry]
    f_names = ["Seed block fitness", "Airspace fitness", "Bounding wall fitness", "Door fitness", "Symmetry Fitness"]
    fitness = {}
    for name, f in zip(f_names, fit_funcs):
        fitness[name] = f(None, input, output)
    return fitness

def fit_bounding_wall(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    # Checks for a complete wall with no airgaps
    non_zero = 0
    for y in output:
        # Get number of air gaps in each layer of the wall
        non_zero += np.count_nonzero(y[0])  
        non_zero += np.count_nonzero(y[:, 0][1:-1])
        non_zero += np.count_nonzero(y[:, -1][1:-1])
        non_zero += np.count_nonzero(y[-1])
    perimeter = 2 * (len(output[0][0]) - 1 + len(output[0]))
    return non_zero / (perimeter * len(output))


def fit_door(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    # Check door exists and is on floor
    # Door ID's
    ids = [93, 98, 277, 278, 279, 280, 281]
    # Catch numpy future warning
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # Iterate over each layer
        for yi, y in enumerate(output):
            # Get non-corner walls
            walls = [y[0][1:-1], y[:, 0][1:-1], y[:, -1][1:-1], y[-1][1:-1]]
            for wall in walls:
                # Check if each door ID exists
                for door in ids:
                    if str(door) in wall:
                        if yi == 0 or yi == 1:
                            return 1
    return 0                
         
                    
def fit_airspace(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    fit = 0
    non_zero = 0
    # Get airspace inside building (removing walls)
    space = output[:-1]
    for z in space:
        for x in z[1:-1]:
            # Count non air blocks
            non_zero += np.count_nonzero(x[1:-1])

    # Get volume of interior space
    vol = (len(output) - 1) * (len(output[0]) - 2) * (len(output[0][0]) - 2)
    # Get percentage of air blocks
    return non_zero / vol


def fit_seed_blocks(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    # Get frequency of blocks being used
    # Check if seed values are in top 8 (top 7 - air) most common blocks (Allow for some randomness)
    seeds = input[3:6]
    counts = []
    
    # Get and format unique counts
    c = np.unique(output, return_counts=True)
    for id, count in zip(c[0], c[1]):
        counts.append((id, count))
    
    # Sort counts
    counts = sorted(counts, key=lambda x: x[1], reverse=True)
    
    # Get top 8
    top = 8
    if len(counts) > top:
        counts = counts[:top+1]
    
    # Get IDs 
    counts = [id for id, __ in counts]
    
    # Get number of seed blocks found in structure
    found = 0
    for seed in seeds:
        if str(seed) in counts:
            found += 1

    # Return percentage of seeds found for fitness
    return found/len(seeds)
    
def fit_symmetry(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
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
        print(a)
        print(b)
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