from src.neatfitnessinterface import RoofInterface, HouseInterface
import numpy as np
import os
from src.Blocks.block_interactions import BlockInterface

block_path = "src/Blocks/blocks.csv"
bi = BlockInterface(block_path, connect=False)

roof_net = RoofInterface(
    config_file="config/Roof-NEAT-config",
    block_path=block_path,
    log_root="logs/roof/",
    overwrite_logs=True,
    n_generations=5000,
    n_input=13,
    n_output=8,
    n_pop=5,
).run()


# house_net = HouseInterface(
#     config_file="config/House-NEAT-config",
#     block_path=block_path,
#     log_root="logs/house/",
#     overwrite_logs=True,
#     n_generations=1000,
#     n_input=26,
#     n_output=len(bi.blocklist)-1,
#     n_pop=5,
# ).run()
