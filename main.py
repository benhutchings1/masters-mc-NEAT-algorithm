from src import neat
from src.Fitness import fitness, generate_building
from src.Blocks import block_interactions
from tests.structure_test import test_structure


config_path = "config/NEAT-config"
blocks_path = "src/Blocks/blocks.csv"
bi = block_interactions.BlockReader(block_path=blocks_path, connect=False)
neat.edit_config(config_path, input_size = 26, pop_size=10, 
                 output_size=len(bi.blocklist))
print(neat.run(config_path, fitness.test_fitness, n_generations=1))
