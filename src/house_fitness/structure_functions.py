import numpy as np
import neat
from typing import List, Dict
import warnings
from functools import lru_cache
from src.logger import StructLogger

def structure_score(genomes:List[neat.DefaultGenome], input:list, outputs:List[np.array], logger:StructLogger) \
    -> List[float]:
    # Setup logger headers
    if logger.first_time:
        setup_logger(logger)
        logger.first_time = False
    # Run score functions
    score_funcs = [score_seed_blocks, score_airspace, score_bounding_wall, score_door, score_symmetry]
    scores = []
    for (__, genome), output in zip(genomes, outputs):
        g_score = []
        for f in score_funcs:
            g_score.append(f(genome, input, output))
        scores.append(np.average(g_score))
        # Log individual values
        logger.log_value(g_score)
    return scores

def setup_logger(logger:StructLogger) -> None:
    f_names = ["Seed_block_score", "Airspace_score", "Bounding_wall_score", "Door_score", "Symmetry_score"]
    logger.add_header(f_names)
    logger.start_gen()

def single_structure_score(input:list, output:np.array) -> Dict:
    score_funcs = [score_seed_blocks, score_airspace, score_bounding_wall, score_door, score_symmetry]
    score_names = ["Seed block score", "Airspace score", "Bounding wall score", "Door score", "Symmetry score"]
    score = {}
    for name, f in zip(score_names, score_funcs):
        score[name] = f(None, input, output)
    return score

def score_bounding_wall(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    # Checks for a complete wall with no airgaps
    non_zero = 0
    for y in output:
        # Get number of air gaps in each layer of the wall
        non_zero += np.count_nonzero(y[0])  
        non_zero += np.count_nonzero(y[:, 0][1:-1])
        non_zero += np.count_nonzero(y[:, -1][1:-1])
        non_zero += np.count_nonzero(y[-1])
    perimeter = output.shape[0] * (output.shape[1] * 2 + (output.shape[2] - 2)*2)
    return non_zero / perimeter


def score_door(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    # Check door exists and is on floor
    # Door ID's
    ids = get_door_ids()
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
                    if door in wall:
                        if yi == 0 or yi == 1:
                            return 1.0
    return 0.0                

@lru_cache(maxsize=None)
def get_door_ids():
    from src.Blocks.block_interactions import BlockInterface
    bi = BlockInterface(block_path="src/Blocks/blocks.csv", connect=False)
    door_ids = ["64", "71", "193", "194", "195", "196", "197"]
    return [bi.blockmap.get(id) for id in door_ids]
    
              
def score_airspace(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    non_zero = 0

    # Get airspace inside building (removing walls)
    for y in output:
        for z in y[1:-1]:
            # Count non air blocks
            non_zero += np.count_nonzero(z[1:-1])
    
    # Get volume of interior space
    vol = ((output.shape[1] - 2) * (output.shape[2] - 2)) * output.shape[0]

    # Get percentage of air blocks
    return 1- (non_zero / vol)


def score_seed_blocks(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
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
        if seed in counts:
            found += 1

    # Return percentage of seeds found for score
    return found/len(seeds)
    
def score_symmetry(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    return max(score_vert_symmetry(output), score_horiz_symmetry(output))

def score_vert_symmetry(output:np.array) -> float:
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


def score_horiz_symmetry(output:np.array) -> float:
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