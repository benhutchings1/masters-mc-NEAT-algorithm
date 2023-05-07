import numpy as np
import neat
from typing import List, Dict
import warnings
from functools import lru_cache

def __get_funcs():
    return (
        [score_seed_blocks, score_symmetry, score_door, score_bounding_wall, score_block_variation],
        ["score_seed_blocks", "score_symmetry", "score_door", "score_bounding_wall", "score_block_variation"]
    )

def structure_score(genomes:List[neat.DefaultGenome], input:list, outputs:List[np.array], logger) \
    -> List[float]:
    # Setup logger headers
    if logger.first_time:
        setup_logger(logger)
        logger.first_time = False
    # Run score functions
    score_funcs, __ = __get_funcs()
    scores = []
    for (__, genome), output in zip(genomes, outputs):
        g_score = []
        for f in score_funcs:
            g_score.append(f(genome, input, output))
        scores.append(np.average(g_score))
        # Log individual values
        logger.log_value(g_score)
    return scores

def setup_logger(logger) -> None:
    logger.add_header(__get_funcs()[1])
    logger.start_gen()

def single_structure_score(input:list, output:np.array, avg=False) -> Dict:
    score_funcs, score_names = __get_funcs()
    if not avg:
        score = {}
        for name, f in zip(score_names, score_funcs):
            score[name] = f(None, input, output) 
    else:
        score = 0
        for f in score_funcs:
            score += f(None, input, output)
        score = score/len(score_funcs)
    return score

def score_bounding_wall(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    return np.count_nonzero(output)/(output.shape[0]*output.shape[1])


def score_door(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    # Check door exists and is on floor
    # Door ID's
    id = get_door_id()
    # Check a door id is on the bottom layer
    if id in output[0]:
        return 1
    return 0

@lru_cache(maxsize=None)
def get_door_id():
    from src.Blocks.block_interactions import BlockInterface
    bi = BlockInterface(block_path="src/Blocks/blocks.csv", connect=False)
    return bi.blockmap.get("64")
    
    
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
    return max(score_vert_symmetry(input, output), score_horiz_symmetry(input, output))

def score_vert_symmetry(input:np.array, output:np.array) -> float:
    length = input[1]
    width = input[2]
    length_mid = __get_midpoint_bound(length)
    symmetry_sum = 0
    # Get both sides across symmetry line    
    boundry = length_mid[1]+2*(length_mid[0]+1)+(width-2) 
    for row in output:
    # RHS of symmetry line
        a = np.array(row[length_mid[1]:boundry])
        # LHS of symmetry line
        b = np.concatenate(
            (np.flip(row[:length_mid[0]+1]),
            np.flip(row[-1*((width-2)+length_mid[0]+1):])))
        symmetry_sum += np.count_nonzero(a==b)
        
    return symmetry_sum / (len(output) * (2*(length_mid[0]+1) + width - 2))


def score_horiz_symmetry(input:np.array, output:np.array) -> float:
    length = input[1]
    width = input[2]
    width_mid = __get_midpoint_bound(width)
    symmetry_sum = 0
    
    for row in output:
        a = np.concatenate((
            row[-1*(width_mid[0]):],
              row[:length+width_mid[0]]
              ))
        b = np.flip(row[length+width_mid[1]-1:len(row)-width_mid[1]+1])
        symmetry_sum += np.count_nonzero(a==b)
    return symmetry_sum / (len(output) * (length + 2*(width_mid[0])))
    

def __get_midpoint_bound(x):
    x = x-2
    if x%2 == 0:
        return (int(x/2), int(x/2)+1)
    else:
        return (int(((x+1)/2)-1), int(((x+1)/2)+1))
    
def score_block_variation(genome:neat.DefaultGenome, input:np.array, output:np.array) -> float:
    unique_blocks = len(np.unique(output, return_counts=True))
    if unique_blocks -1 <= 3:
        return unique_blocks / 3
    else:
        return 0.75