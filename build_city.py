from src.Blocks.city_builder import CityBuilder
from src.Blocks.block_interactions import BlockInterface
import numpy as np
from src.house_fitness.structure_functions import score_block_variation
import random
from src.visualise import multi_struct_plot
import os
import skimage.morphology as morph  
from src.Blocks import no_comm_gen

house_config = "config/House-NEAT-config"
roof_config = "config/Roof-NEAT-config"
block_path = "src/Blocks/blocks.csv"

builders = {}

run_config = {
    "high_novelty":[["22974", "22974"],["21979", "2999"]],
    "all_novelty":[["2999", "2999"],["2999", "2999"]],
    "control":[["22974", "22974"],["21979", "21979"]],
    "low_novelty":[["22974", "22974"],["21979", "21979"]]
}


for source in ["high_novelty", "all_novelty", "control", "low_novelty"]:
    checkpoints = run_config[source]
    houses = [
        [f"data/{source}/iter1/house/checkpoints/", f"NEAT-checkpoint-{checkpoints[0][0]}"],
        [f"data/{source}/iter2/house/checkpoints/", f"NEAT-checkpoint-{checkpoints[0][1]}"]
    ]
    roofs = [
        [f"data/{source}/iter1/roof/checkpoints", f"NEAT-checkpoint-{checkpoints[1][0]}"],
        [f"data/{source}/iter2/roof/checkpoints", f"NEAT-checkpoint-{checkpoints[1][1]}"]
    ]


    cb = CityBuilder(block_path="src/Blocks/blocks.csv", house_config=house_config, roof_config=roof_config, 
                    log_path="temp/", filename="multilog.txt", overwrite_log=True, connect=False)
    builders[source] = cb

    # Read in populations
    for (h_path, h_check),(r_path, r_check) in zip(houses, roofs): 
        cb.read_in_pop(config_path=house_config,checkpoint_path=h_path, checkpoint_name=h_check)
        cb.read_in_pop(config_path=roof_config, checkpoint_path=r_path, checkpoint_name=r_check, house=False)

    cb.place_city(iterations=200, plot_size=15, orientation="N", x0=0, z0=0, gap=2)

# Results analysis
pl = multi_struct_plot.MultiStructurePlot()
pl.plot_bar_var(builders, "out/", "variance_comparison.png")
pl.plot_gen_time_bar(builders, "out/", "times.png")
pl.plot_block_frequencies(builders, "out/", "block_freq.png")
pl.plot_all_scores(builders, "out/", ["house_scores.png", "roof_scores.png"])