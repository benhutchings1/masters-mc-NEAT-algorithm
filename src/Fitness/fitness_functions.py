import numpy as np
import neat
from typing import List

# Check for windows
# Check for interior space
# Check percentage of interior space made of air
# Check has a door on low floor
# Check for slanted roof? or flat roof?

def structure_fitness(genomes:List[neat.DefaultGenome], input:np.array, outputs:List[np.array]) \
    -> List[float]:
    fit_funcs = [fit_airspace, fit_bounding_wall, fit_door]
    for (gid, genome), output in zip(genomes, outputs):
        g_fit = []
        for f in fit_funcs:
            g_fit.append(f(genome, input, output))
        print(g_fit)
        break   

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
    # Get airspace inside building
    space = output[:-1]
    for z in space:
        for x in z[1:-1]:
            non_zero += np.count_nonzero(x[1:-1])
    vol = (len(output) - 1) * (len(output[0]) - 2) * (len(output[0][0]) - 2)
    return 1 - (non_zero / vol)
    