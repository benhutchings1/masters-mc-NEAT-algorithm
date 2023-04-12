from src import neat
from src.Fitness import fitness, generate_building
from src.Blocks import block_interactions
from tests.structure_test import test_structure
import numpy as np


config_path = "config/NEAT-config"
blocks_path = "blocks.csv"
neat.edit_config(config_path, input_size = 26, pop_size=20)
print(neat.run(config_path, fitness.test_fitness, n_generations=1000))
