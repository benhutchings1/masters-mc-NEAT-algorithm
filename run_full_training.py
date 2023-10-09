from src.neatfitnessinterface import RoofInterface, HouseInterface
from src.Blocks.block_interactions import BlockInterface
import CONFIG

# ============== EXPERIMENT CONFIGURATION ============== 
# Model config
CONFIG_ROOF_EXP = {                             # Roof Model
    "config_file": "config/Roof-NEAT-config",   # NEAT roof model config file (default: config/Roof-NEAT-config)
    "n_generations": 5,                         # Number of generations to run roof model
    "n_output": 8,                              # The max roof height for the building
    "n_pop": 50,                                # Model population size
    "checkpoint_rate": 1                        # Make a population checkpoint (for recovery and testing) every n generations
}
CONFIG_STRUCT_EXP = {                           # Structure Model
    "config_file": "config/House-NEAT-config",  # NEAT structure model config file (default: config/House-NEAT-config)
    "n_generations": 5,                         # Number of generations to run structure model
    "n_pop": 10,                                # Model population size
    "checkpoint_rate": 1                        # Make a population checkpoint (for recovery and testing) every n generations
}

# Experiment specific configs
# 3 experiment types, each run for 2 iterations
# Experiment types:
#       Control (no novelty)
#       Low Novelty 
#       High Novelty
# Output location for generated data retrieved from CONFIG.py
out = CONFIG.DATA_OUTPUT_PATH
iters = [
    [f"{out}/control/iter1/", {"use_novelty":False}],
    [f"{out}/control/iter2/", {"use_novelty":False}],
    [f"{out}/high_novelty/iter1/", {"use_novelty":True, "novelty_ratio":(1, 4)}],
    [f"{out}/high_novelty/iter2/", {"use_novelty":True, "novelty_ratio":(1, 4)}],
    [f"{out}/low_novelty/iter1/", {"use_novelty":True, "novelty_ratio":(4, 1)}],
    [f"{out}/low_novelty/iter2/", {"use_novelty":True, "novelty_ratio":(4, 1)}],
]

# ============== RUN EXPERIMENTS ============== 
# Define block interface
bi = BlockInterface(CONFIG.BLOCK_PATH, connect=CONFIG.SERVER_CONNECT)
# Iterate over all experimentation parameters
for i, (path, parameters) in enumerate(iters):
    print()
    print("=================================================")
    print(f"Experiment {i}")
    print(f"Training roof model for experiment {i}")
    roof_net = RoofInterface(
        # Config file path
        config_file=CONFIG_ROOF_EXP["config_file"],
        # Path to block descriptors
        block_path=CONFIG.BLOCK_PATH,
        # Path to log files root
        log_root=path+"roof/",
        # Choose to overwrite logs instead of appending
        overwrite_logs=True,
        # Training parameters
        n_generations=CONFIG_ROOF_EXP["n_generations"],
        n_input=13,
        n_output=CONFIG_ROOF_EXP["n_output"], 
        n_pop=CONFIG_ROOF_EXP["n_pop"],
        # How often checkpoints are taken
        checkpoint_rate=CONFIG_ROOF_EXP["checkpoint_rate"],
        # Experimentation specific parameters
        **parameters
    ).run()
    # To restart training from an existing checkpoint use
    # .run_from_checkpoint(path_to_checkpoint)
    # instead of .run()
    
    print()
    print("=================================================")
    print()
    
    print(f"Training house model for experiment {i}")
    house_net = HouseInterface(
        # Config file path
        config_file=CONFIG_STRUCT_EXP["config_file"],
        # Path to block descriptors
        block_path=CONFIG.BLOCK_PATH,
        # Path to log files root
        log_root=path+"house/",
        # Choose to overwrite logs instead of appending
        overwrite_logs=True,
        # Training parameters
        n_generations=CONFIG_STRUCT_EXP["n_generations"],
        n_input=16,
        n_pop=CONFIG_STRUCT_EXP["n_pop"],
        # Model output size fixed to no. blocks
        n_output=len(bi.blocklist)-1,
        # How often checkpoints are taken
        checkpoint_rate=CONFIG_STRUCT_EXP["checkpoint_rate"],
        # Experimentation specific parameters
        **parameters
    ).run()
    # To restart training from an existing checkpoint use
    # .run_from_checkpoint(path_to_checkpoint)
    # instead of .run()
    print("=================================================")
    print()