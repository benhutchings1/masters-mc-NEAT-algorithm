import numpy as np
import neat
from typing import List

# identify bounding wall
# Check wall has no airgaps
# Check for windows
# Check for interior space
# Check percentage of interior space made of air
# Check has a door on low floor
# Check for slanted roof? or flat roof?
# Check for diversity

def structure_fitness(genomes:List[neat.DefaultGenome], input:np.array, outputs:List[np.array]) \
    -> List[float]:
    fit_funcs = [fit_bounding_wall]
    for (gid, genome), output in zip(genomes, outputs):
        for f in fit_funcs:
            f(genome, input, output)
        break   

def fit_bounding_wall(genome:neat.DefaultGenome, input:np.array, output:np.array):
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

