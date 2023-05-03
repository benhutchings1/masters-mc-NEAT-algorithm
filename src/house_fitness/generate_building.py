import numpy as np
import neat
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_generate(nets:neat.nn.FeedForwardNetwork, input_config:List[float]) -> tuple:
    # Get genome information
    net_results = [None] * (len(nets) + 1)
    
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
    # Get parameters
    height = input_config[0]
    length = input_config[1]
    width = input_config[2]
    # Get input template
    template_input = np.zeros((8 + 2 + len(input_config))).astype(int)
    template_input[:len(input_config)] = input_config
    # Get output template
    out = np.zeros((height, (2*length) + (2*width) - 4)).astype(int)
    
    # Fill output using model
    for yi, y in enumerate(out):
        for xi, x in enumerate(y):
            # Format input to network
            template_input[len(input_config)] = yi
            template_input[len(input_config) + 1] = xi
            template_input[len(input_config) + 2:] = get_surrounding(out, yi, xi)
            # Pass input to network
            out[yi][xi] = np.argmax(net.activate(template_input))
    print(out)
    return genome_id, out
    

def get_surrounding(arr, yi, xi):
    out = np.zeros((8)).astype(int)
    idx_count = 0
    for i in range(3):
        for j in range(3):
            y = yi-1+i
            x = xi-1+j
            if not y < 0 and not y > len(arr) - 1:
                if not x < 0 and not x > len(arr[0]) - 1:
                    if i == 1 and j == 1:   
                        continue
                    out[idx_count] = arr[y][x]
                    idx_count += 1
    return np.array(out)
