from . import generate_building, fitness_functions, novelty
import neat
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import random

def test_fitness(genomes:list, config:neat.Config):
    """
    Main function called to evaluate a set of genomes
    Requires a neat.Config object
    Returns a list of fitness values for each genome
    """
    # Configuration options
    HEIGHT = 5
    LENGTH = 5
    WIDTH = 5
    
    seeds = random.sample(range(0, 307), 3)

    input_config = [
        HEIGHT,
        LENGTH,
        WIDTH,
        seeds[0],
        seeds[1],
        seeds[2]
    ]

    # Get genome information
    gids = [g[0] for g in genomes]
    nets = [net for net in __generate_nets(genomes, config)]
    net_results = [None] * (len(nets) + 1)

    # Generate one building using each genome
    # Split processing over all CPU's
    with ProcessPoolExecutor() as exe:
        futures = [
            exe.submit(generate_building.generate, gid, net, input_config)
                        for gid, net in zip(gids, nets)
                ]
        # Process results once completed
        for result in as_completed(futures):
            result = result.result()
            net_results[result[0] - 1] = result[1]

    
    # Evaluate Fitness for model's output
    fitnesses = combined_fitness_tests(genomes, input_config, net_results)
    raise NotImplementedError

    

def __generate_nets(genomes:list, config:neat.Config) -> list:
    """
    Uses each genome to make a list of corresponding neat.FeedForwardNetwork's
    returns list of networks
    """
    nets = []
    for genome_id, genome in genomes:
        # Create net from given genome
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
    return nets
        

def combined_fitness_tests(genomes:neat.DefaultGenome, input:np.array, outputs:np.array) -> int:
    """
    Combines all fitness tests
    """
    # novelty.novelty_fitness(genomes)
    fitness_functions.structure_fitness(genomes, input, outputs)
    raise NotImplementedError