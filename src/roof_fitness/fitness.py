from . import generate_roof, structure_functions
from src import novelty as nvlty, logger
from src.Blocks import block_interactions
import neat
import numpy as np
import random
import math

class Fitness:
    def __init__(self, block_path, novelty_log_path, struct_log_path, overwrite=True) -> None:
        # Setup novelty calculator with logging
        self.novelty = nvlty.Novelty(novelty_log_path, overwrite=overwrite)
        # Setup block interactor
        self.bi = block_interactions.BlockInterface(block_path=block_path, connect=False)
        # Setup structural data logger
        self.struct_logger = logger.StructLogger(struct_log_path, "struct_log.csv", overwrite_log=overwrite)
        
    def __call__(self, genomes:list, config:neat.Config, return_best=False):
        """
        Main function called to evaluate a set of genomes
        Requires a neat.Config object
        return_best: returns the best individual from genomes
        Returns a list of fitness values for each genome
        """
        # Configuration options
        HEIGHT = random.randint(2, 5)
        LENGTH = random.randint(5, 10)
        WIDTH = random.randint(5, 10)
        
        # Generate 3 random blocks
        seeds = random.sample(range(0, len(self.bi.blocklist) - 1), 3)

        # Format inputs
        input_config = [
            HEIGHT,
            LENGTH,
            WIDTH
        ]

        nets = [net for net in self.__generate_nets(genomes, config)]
        net_results = generate_roof.parallel_generate(nets, input_config)
        # Evaluate Fitness for model's output
        best_fit = 0
        best_genome = 0
        
        for (__, genome), fit in zip(genomes, self.fitness_test(genomes, input_config, net_results)):
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
        Uses each genome to make a list of corresponding neat.FeedForwardNetwork neural networks
        returns list of networks
        """
        nets = []
        for __, genome in genomes:
            # Create net from given genome
            nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        return nets
            

    def fitness_test(self, genomes:neat.DefaultGenome, input:np.array, outputs:np.array) -> int:
        """
        Gets fitness of each genome
        """
        # Start logging generation
        self.struct_logger.start_gen()
        
        # Compute ratio of structure score to novelty
        r_nov, r_score = 1, 2 # r_nov:r_score
        p_nov = r_nov/(r_nov+r_score)
        p_score = r_score/(r_nov+r_score)
        
        # Compute structure score and novelty then combine to make fitness score
        out = []
        novelty = self.novelty.novelty_score(genomes, math.ceil(len(genomes)/2))
        score = structure_functions.structure_score(input, outputs, logger=self.struct_logger) 
        for ni, sc in zip(novelty, score):
            out.append(ni * p_nov + sc * p_score)
        return out