from . import generate_building
from . import fitness_functions
from .. import novelty as nvlty
from src.Blocks import block_interactions
from src.logger import StructLogger
import neat
import numpy as np
import random
import math

class Fitness:
    def __init__(self, block_path, novelty_log_path, struct_log_path, overwrite=True) -> None:
        self.novelty = nvlty.Novelty(novelty_log_path, overwrite=overwrite)
        self.BI = block_interactions.BlockInterface(block_path=block_path, connect=False)
        self.struct_logger = StructLogger(log_path=struct_log_path, filename="struct_log.csv", overwrite_log=overwrite)
        
    def __call__(self, genomes:list, config:neat.Config, return_best=False):
        """
        Main function called to evaluate a set of genomes
        Requires a neat.Config object
        return_best: returns the best individual from genomes
        Returns a list of fitness values for each genome
        """
        # Configuration options
        HEIGHT = random.randint(2, 7)
        LENGTH = random.randint(5, 10)
        WIDTH = random.randint(5, 10)
        
        # Generate 3 random blocks
        seeds = random.sample(range(0, len(self.BI.blocklist) - 1), 3)

        # Format inputs
        input_config = [
            HEIGHT,
            LENGTH,
            WIDTH,
            seeds[0],
            seeds[1],
            seeds[2]
        ]

        nets = [net for net in self.__generate_nets(genomes, config)]
        net_results = generate_building.parallel_generate(nets, input_config)
        # Evaluate Fitness for model's output
        best_fit = 0
        best_genome = 0
        
        # Get fitness
        for (__, genome), fit in zip(genomes, self.combined_fitness_tests(genomes, input_config, net_results)):
            # Get best model
            if return_best:
                if fit > best_fit:
                    best_fit = fit
                    best_genome = genome
        
            genome.fitness = fit

        if best_fit:
            return best_genome
        
    def __generate_nets(self, genomes:list, config:neat.Config) -> list:
        """
        Uses each genome to make a list of corresponding neat.FeedForwardNetwork's
        returns list of networks
        """
        nets = []
        for __, genome in genomes:
            # Create net from given genome
            nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        return nets
            

    def combined_fitness_tests(self, genomes:neat.DefaultGenome, input:np.array, outputs:np.array) -> int:
        """
        Combines all fitness tests
        """
        # Start new logging generation
        self.struct_logger.start_gen()
         
        # Compute novelty and fitness and take ratio
        r_nov, r_fit = 1, 2 # r_nov:r_fit
        out = []
        novelty = self.novelty.novelty_fitness(genomes, math.ceil(len(genomes)/2))
        fitness = fitness_functions.structure_fitness(genomes, input, outputs, logger=self.struct_logger) 
        for ni, fi in zip(novelty, fitness):
            out.append(ni * (r_nov/(r_nov+r_fit)) + fi * (r_fit/(r_nov+r_fit)))
        return out