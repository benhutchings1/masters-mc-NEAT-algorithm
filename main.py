from src import neat
from src.Fitness import fitness
from src.Blocks import block_interactions


config_path = "config/NEAT-config"
blocks_path = "blocks.csv"
neat.edit_config(config_path, input_size = 26, pop_size=10)
neat.run(config_path, fitness.test_fitness, n_generations=100)
