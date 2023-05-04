from src.neatfitnessinterface import RoofInterface, HouseInterface
from src.Blocks.block_interactions import BlockInterface
from typing import List
import neat
import random
import numpy as np
from src.house_fitness import structure_functions
from src.house_fitness import generate_building

block_path = "src/Blocks/blocks.csv"
bi = BlockInterface(block_path, connect=False)

iters = [
    ["data/control/iter1/", {"use_novelty":False}],
    ["data/control/iter2/", {"use_novelty":False}],
    ["data/high_novelty/iter1/", {"use_novelty":True, "novelty_ratio":(1, 4)}],
    ["data/high_novelty/iter2/", {"use_novelty":True, "novelty_ratio":(1, 4)}],
    ["data/low_novelty/iter1/", {"use_novelty":True, "novelty_ratio":(3, 4)}],
    ["data/low_novelty/iter2/", {"use_novelty":True, "novelty_ratio":(3, 4)}],
    ["data/dynamic/iter1/", {"use_novelty":True, "use_dynamic_novelty":True}],
    ["data/dynamic/iter2/", {"use_novelty":True, "use_dynamic_novelty":True}],
]


for path, par in iters:
    print(par)    
    roof_net = RoofInterface(
        config_file="config/Roof-NEAT-config",
        block_path=block_path,
        log_root=path+"roof/",
        overwrite_logs=True,
        n_generations=1,
        n_input=13,
        n_output=8,
        n_pop=100,
        checkpoint_rate=5,
        **par
    ).run()

    house_net = HouseInterface(
        config_file="config/House-NEAT-config",
        block_path=block_path,
        log_root=path+"house/",
        overwrite_logs=True,
        n_generations=1,
        n_input=16,
        n_output=len(bi.blocklist)-1,
        n_pop=3,
        checkpoint_rate=5,
        **par
    ).run()
    break