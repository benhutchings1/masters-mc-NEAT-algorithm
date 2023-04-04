from src import neat
from src.Fitness.fitness import Fitness
from src.Blocks import block_interactions


config_path = "config/NEAT-config"
blocks_path = "blocks.csv"

neat.run(config_path, Fitness.calc_fitness)
