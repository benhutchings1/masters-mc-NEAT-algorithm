from src import neat
from src.Fitness import fitness
from src.Blocks import block_interactions


config_path = "config/NEAT-config"
blocks_path = "blocks.csv"
neat.config_inputs(config_path, 23)
neat.run(config_path, fitness.test_fitness, n_generations=1)
