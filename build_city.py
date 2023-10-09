from src.Blocks.city_builder import CityBuilder
from src.visualise import multi_struct_plot
import CONFIG
import os

city_data_out = CONFIG.VISUALISATION_OUTPUT + "/city"

# Automatically get highest trained models
def get_models(data_root):
    def get_biggest_dir(path):
        # Get version number
        files = [int(file[16:]) for file in os.listdir(path)]
        # Get highest version
        return max(files)
    out = {}
    for src in ["control", "high_novelty", "low_novelty"]:
        out[src] = [["1", "1"],["1", "1"]]
        for n_iter, iter in enumerate(["iter1", "iter2"]):
            for n_mod, mod in enumerate(["house", "roof"]):
                # Format path
                loc = f"{CONFIG.DATA_OUTPUT_PATH}/{src}/{iter}/{mod}/checkpoints"
                # Save biggest
                out[src][n_iter][n_mod] = str(get_biggest_dir(loc))
    return out

# Automatically get model version
run_config = get_models(CONFIG.DATA_OUTPUT_PATH)

# Specify which generation of model to use
# Update depending on how far into training got
# Will be automatically found but can be overriden by changing if statement
if False:
    run_config = {
        # model type: [[house_iter1, roof_iter1],[house_iter2, roof_iter2]]
        "high_novelty":[["1", "1"],["1", "1"]],
        "control":[["1", "1"],["1", "1"]],
        "low_novelty":[["1", "1"],["1", "1"]]
    }


builders = {}
# Get all models from saved models
for source in ["high_novelty", "control", "low_novelty"]:
    checkpoints = run_config[source]
    houses = [
        [f"{CONFIG.DATA_OUTPUT_PATH}/{source}/iter1/house/checkpoints/", f"NEAT-checkpoint-{checkpoints[0][0]}"],
        [f"{CONFIG.DATA_OUTPUT_PATH}/{source}/iter2/house/checkpoints/", f"NEAT-checkpoint-{checkpoints[0][1]}"]
    ]
    roofs = [
        [f"{CONFIG.DATA_OUTPUT_PATH}/{source}/iter1/roof/checkpoints", f"NEAT-checkpoint-{checkpoints[1][0]}"],
        [f"{CONFIG.DATA_OUTPUT_PATH}/{source}/iter2/roof/checkpoints", f"NEAT-checkpoint-{checkpoints[1][1]}"]
    ]


    cb = CityBuilder(block_path=CONFIG.BLOCK_PATH, house_config=CONFIG.HOUSE_CONFIG_PATH, roof_config=CONFIG.ROOF_CONFIG_PATH, 
                    log_path=city_data_out, filename="multilog.txt", overwrite_log=True, connect=CONFIG.SERVER_CONNECT)
    builders[source] = cb

    # Read in populations
    for (h_path, h_check),(r_path, r_check) in zip(houses, roofs): 
        cb.read_in_pop(config_path=CONFIG.HOUSE_CONFIG_PATH,checkpoint_path=h_path, checkpoint_name=h_check)
        cb.read_in_pop(config_path=CONFIG.ROOF_CONFIG_PATH, checkpoint_path=r_path, checkpoint_name=r_check, house=False)

    print(f"==== Using {source} models ====")
    cb.place_city(iterations=CONFIG.CITY_GENERATION_SIZE, plot_size=15, orientation="N", x0=1, z0=1, gap=2)

# Results analysis
pl = multi_struct_plot.MultiStructurePlot()
pl.plot_bar_var(builders, city_data_out+"/", "variance_comparison.png")
pl.plot_gen_time_bar(builders, city_data_out+"/", "times.png")
pl.plot_block_frequencies(builders, city_data_out+"/", "block_freq.png")
pl.plot_all_scores(builders, city_data_out+"/", ["house_scores.png", "roof_scores.png"])    