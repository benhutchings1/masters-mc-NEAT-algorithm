import numpy as np
import neat
from typing import List

# Check for windows
# Check for slanted roof? or flat roof?
# Size of model?

# Fitness 0 = low, 1 = high

def structure_fitness(genomes:List[neat.DefaultGenome], input:np.array, outputs:List[np.array]) \
    -> List[float]:
    fit_funcs = [fit_seed_blocks, fit_airspace, fit_bounding_wall, fit_door]
    fitnesses = []
    for (gid, genome), output in zip(genomes, outputs):
        g_fit = []
        for f in fit_funcs:
            g_fit.append(f(genome, input, output))
        fitnesses.append(np.average(g_fit))
    return fitnesses


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
    ids=  [297, 298, 304, 305, 306, 307]
    fit = 0
    # Iterate over each layer
    for yi, y in enumerate(output):
        # Get non-corner walls
        walls = [y[0][1:-1], y[:, 0][1:-1], y[:, -1][1:-1], y[-1][1:-1]]
        for wall in walls:
            # Check if each door ID exists
            for door in ids:
                if door in wall:
                    if yi == 0 or yi == 1:
                        fit = 1
                        return fit
    return fit                
         
                    
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
    return 1 - (non_zero / vol)


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

    # Get top 6
    top = 8
    if len(counts) > top:
        counts = (counts[0][:top], counts[1][:top])

    # Get IDs 
    counts = [id for id, __ in counts]
    # Get number of seed blocks found in structure
    found = 0
    for seed in seeds:
        if seed in counts:
            found += 1

    # Return percentage of found for fitness
    return found/len(seeds)
    
    
    
    