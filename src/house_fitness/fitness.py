from . import generate_building
from . import structure_functions
from .. import novelty as nvlty
from src.Blocks import block_interactions
from src.logger import StructLogger
import neat
import numpy as np
import random
import math

class Fitness:
    def __init__(self, block_path, novelty_log_path, struct_log_path, overwrite=True, use_novelty=True, use_dynamic_novelty=False, novelty_ratio=None, squash_function=True) -> None:
        self.novelty = nvlty.Novelty(novelty_log_path, overwrite=overwrite, squash_function=squash_function)
        self.BI = block_interactions.BlockInterface(block_path=block_path, connect=False)
        self.struct_logger = StructLogger(log_path=struct_log_path, filename="struct_log.csv", overwrite_log=overwrite)

        #  Default to no novelty (control)
        self.nov_tech = 0
                     
        if use_novelty:
            # Dynamic novelty or fixed novelty
            if use_dynamic_novelty:
                # Save novelty type
                self.nov_tech = 1
                # Init dynamic novelty object
                self.dn = nvlty.DynamicNovelty()
            else:
                # Check if novelty ratio is defined
                if novelty_ratio is None:
                    raise ValueError("No novelty technique specified")
                # Get novelty ratio
                assert type(novelty_ratio) == tuple and len(novelty_ratio) == 2
                self.r_nov = novelty_ratio[0]
                self.r_score = novelty_ratio[1]
                # Save novelty type
                self.nov_tech = 2
            
        
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
        seeds = random.sample(range(1, len(self.BI.blocklist) - 1), 3)

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
        Uses each genome to make a list of corresponding neat.FeedForwardNetwork's
        returns list of networks
        """
        nets = []
        for __, genome in genomes:
            # Create net from given genome
            nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        return nets
            

    def fitness_test(self, genomes:neat.DefaultGenome, input:np.array, outputs:np.array) -> int:
        """
        Gets fitness for each genome
        """
        # Start new logging generation
        self.struct_logger.start_gen()
        
        novelty_scores = self.novelty.novelty_score(genomes, math.ceil(len(genomes)/2))
        struct_scores = structure_functions.structure_score(genomes, input, outputs, logger=self.struct_logger) 
        
        # No novelty (control)
        if self.nov_tech == 0:
            return struct_scores
        else:
            if self.nov_tech == 1:   
                return self.dn.get_ratios(novelty_scores, struct_scores)   
            else:
                # STATIC ratio
                # Compute percentage of novelty:structure scores
                p_nov = self.r_nov / (self.r_nov + self.r_score)
                p_score = self.r_score / (self.r_nov + self.r_score)
                # Take percentages of each score as fitness
                fitness = []
                for ns, ss in zip(novelty_scores, struct_scores):
                    fitness.append(ns * p_nov + ss * p_score)
                return fitness