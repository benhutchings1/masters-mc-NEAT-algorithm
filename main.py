from src import neat
from src.Fitness import fitness, generate_building
from src.Blocks import block_interactions
import numpy as np


config_path = "config/NEAT-config"
blocks_path = "blocks.csv"
# neat.edit_config(config_path, input_size = 26, pop_size=10)
# neat.run(config_path, fitness.test_fitness, n_generations=100)
# print("Modelling")
# best_model = neat.checkpoint_best_genome("checkpoints/NEAT-checkpoint-99", config_path, True)
# input = [10, 10, 10, 2, 3, 4]
# print("Block reader")
bi = block_interactions.BlockReader(blocks_path)
# print("Generating")
# blocks = generate_building.generate(1, best_model, input)
# print(blocks)
blocks = np.array([[[2,2],[2,2]]])
bi.place_blocks_np(blocks)


