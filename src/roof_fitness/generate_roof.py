import numpy as np
import neat
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

def parallel_generate(nets:neat.nn.FeedForwardNetwork, input_config:List[float]) -> tuple:
    # Get genome information
    net_results = [None] * (len(nets))
    # Generate one building using each genome
    # Split processing over all CPU's
    with ProcessPoolExecutor() as exe:
        # Send function call to be parallalised
        # Give index to map back to
        futures = [
            exe.submit(generate, i, net, input_config)
                        for i, net in enumerate(nets)
                ]
        # Process results once completed
        for result in as_completed(futures):
            # Get result from future when completed
            result = result.result()
            # Map results back to array
            net_results[result[0]] = result[1]
    return net_results

def generate(genome_id:int, net:neat.nn.FeedForwardNetwork, input_config:List[float]):
    """
    Uses a given model to create a heightmap for a roof to the specified height, length, width
    Get the model to predict a block by giving the surrounding blocks
    
    returns a length x width sized 2d array of values generated by net 
    """
    height = input_config[0]
    length = input_config[1]
    width = input_config[2]
    inputs = np.zeros((13))
    inputs[0] = height
    inputs[1] = length
    inputs[2] = width

    out = np.zeros((length, width)).astype(int)
    # Initialise array randomly to add some randomness
    out.fill(random.randint(0, height - 1))
    for l in range(length):
        for w in range(width):
            surr_points = get_surrounding(out, l, w)
            inputs[3] = l
            inputs[4] = w
            inputs[5:] = surr_points
            # Use model to predict current point
            out[l][w] = np.argmax(net.activate(inputs))
        
    return genome_id, out

def get_surrounding(heightmap, yi, xi):
    out = np.zeros((8)).astype(int)
    idx_count = 0
    for i in range(3):
        for j in range(3):
            y = yi-1+i
            x = xi-1+j
            if not y < 0 and not y > len(heightmap) - 1:
                if not x < 0 and not x > len(heightmap[0]) - 1:
                    if i == 1 and j == 1:   
                        continue
                    out[idx_count] = heightmap[y][x]
                    idx_count += 1
    return np.array(out)