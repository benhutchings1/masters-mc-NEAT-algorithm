from . import generate_building
import neat
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

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

    # Get genome information
    gids = [g[0] for g in genomes]
    nets = [net for net in __generate_nets(genomes, config)]
    net_results = [None] * (len(nets) + 1)

    # Generate one building using each genome
    # Split processing over all CPU's
    with ProcessPoolExecutor() as exe:
        futures = [
            exe.submit(generate_building.generate, gid, net, HEIGHT, LENGTH, WIDTH)
                        for gid, net in zip(gids, nets)
                ]
        # Process results once completed
        for result in as_completed(futures):
            result = result.result()
            net_results[result[0]] = result[1]
    # Evaluate Fitness for model's output
    fitnesses = [
        combined_fitness_test(gid, genome, output) 
        for gid, genome, output in zip(gids, nets, net_results)
    ]

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
        

def combined_fitness_test(gid:int, genome:neat.DefaultGenome, output:np.array) -> int:
    """
    Combines all fitness tests
    """
    return 0