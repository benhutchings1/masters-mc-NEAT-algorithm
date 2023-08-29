from src.neatfitnessinterface import RoofInterface, HouseInterface
from src.Blocks.block_interactions import BlockInterface
from typing import List
import neat
import random
import numpy as np
from src.house_fitness import structure_functions
from src.house_fitness import generate_building

# Output location for generated data
out = "data"

# Establish block connection
block_path = "src/Blocks/blocks.csv"
bi = BlockInterface(block_path, connect=True)

# Experiment configs
iters = [
    [f"{out}/control/iter1/", {"use_novelty":False}],
    [f"{out}/control/iter2/", {"use_novelty":False}],
    [f"{out}/high_novelty/iter1/", {"use_novelty":True, "novelty_ratio":(1, 4)}],
    [f"{out}/high_novelty/iter2/", {"use_novelty":True, "novelty_ratio":(1, 4)}],
    [f"{out}/low_novelty/iter1/", {"use_novelty":True, "novelty_ratio":(4, 1)}],
    [f"{out}/low_novelty/iter2/", {"use_novelty":True, "novelty_ratio":(4, 1)}],
]

# Iterate over all experimentation parameters
for i, (path, parameters) in enumerate(iters):
    print(f"Experiment {i}")
    print("Training roof model")
    roof_net = RoofInterface(
        # Config file path
        config_file="config/Roof-NEAT-config",
        # Path to block descriptors
        block_path=block_path,
        # Path to log files root
        log_root=path+"roof/",
        # Choose to overwrite logs instead of appending
        overwrite_logs=True,
        # Training parameters
        n_generations=1,
        n_input=13,
        n_output=8, # Max roof height 
        n_pop=100,
        # How often checkpoints are taken
        checkpoint_rate=5,
        # Experimentation specific parameters
        **parameters
    ).run()
    print("Training house model")
    house_net = HouseInterface(
        # Config file path
        config_file="config/House-NEAT-config",
        # Path to block descriptors
        block_path=block_path,
        # Path to log files root
        log_root=path+"house/",
        # Choose to overwrite logs instead of appending
        overwrite_logs=True,
        # Training parameters
        n_generations=1,
        n_input=16,
        n_pop=3,
        # Model output size fixed to no. blocks
        n_output=len(bi.blocklist)-1,
        # How often checkpoints are taken
        checkpoint_rate=5,
        # Experimentation specific parameters
        **parameters
    ).run()
