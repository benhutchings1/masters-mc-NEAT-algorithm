from . import generate_building
import neat

def test_fitness(genomes, config:neat.Config,):
    # Iterate through genomes and corresponding networks
    for genome, net in zip(genomes, generate_nets(genomes, config)):
        print(generate_building.generate(net, 15, 15, 15))
        raise NotImplementedError

def generate_nets(genomes, config) -> list:
    nets = []
    for genome_id, genome in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
    return nets
        
