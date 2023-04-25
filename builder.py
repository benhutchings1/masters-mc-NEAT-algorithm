from src.neat import Neat
from src.Blocks import block_interactions
import numpy as np
from src.roof_fitness import generate_roof, fitness as roof_fit
from src.house_fitness import generate_building, fitness as struct_fit

class CheckpointPlacer:
    def __init__(self, x, y, z, type, config_path, block_path,
                checkpoint_path):
        self.x = x
        self.y = y
        self.z = z
        self.type = type
        self.config_path = config_path
        self.block_path = block_path
        self.checkpoint_path = checkpoint_path
        
    def place_checkpoint(self, fitness, checkpoint_name, height, length, width, seeds=None):
        self.nt = Neat(self.config_path, self.block_path, self.checkpoint_path, "temp/",fitness)
        self.nt.connect()
        model = self.nt.checkpoint_best_genome(checkpoint_name, as_model=True)

        if self.type == 1:
            self.roof_placer(model, height, length, width)
        else:
            if seeds is None or type(seeds) is not list:
                raise ValueError
            else:
                self.house_placer(model, height, length, width, seeds)
                
            
    def roof_placer(self, model, height, length, width):
        blocks = self.nt.convert_heightmap(
                generate_roof.generate(0, model, [height, length, width])[1]
            )
        self.nt.place_blocks_np(blocks, x0=self.x, y0=self.y, z0=self.z,
                                isblocklist=True)
        

    def house_placer(self, model, height, length, width, seeds):
        blocks = generate_building.generate(0, model,
                                            [height, length, width]+seeds)[1]
        self.nt.place_blocks_np(blocks, x0=self.x, y0=self.y, z0=self.z,
                                isblocklist=True)
    
t = 1
x = 25
block_path = "src/Blocks/blocks.csv"
bi = block_interactions.BlockInterface(block_path, connect=False)

if t == 1:   
    fit = roof_fit.Fitness(block_path, "temp", "temp")
    cp = CheckpointPlacer(x, -50, 1, 1, "config/Roof-NEAT-config",
        block_path,"logs/roof/checkpoints")
    checkpoints = ["NEAT-checkpoint-4", "NEAT-checkpoint-199", "NEAT-checkpoint-644", "NEAT-checkpoint-934"]
    for i, check in enumerate(checkpoints):
        cp.place_checkpoint(
            fit,
            check,
            2, 6, 6    
        )
        cp.z += 9
    
else:
    fit = struct_fit.Fitness(block_path, "temp", "temp")
    cp = CheckpointPlacer(x, -60, 1, 2, "config/House-NEAT-config",
        block_path,"logs/house/checkpoints")

    checkpoints = ["NEAT-checkpoint-4", "NEAT-checkpoint-24", "NEAT-checkpoint-49", "NEAT-checkpoint-69"]
    seeds = [12, 13, 40]
    print("Using seed blocks")
    print(bi.blocklist[seeds[0]])
    print(bi.blocklist[seeds[1]])
    print(bi.blocklist[seeds[2]])
    for i, check in enumerate(checkpoints):
        cp.place_checkpoint(
            fit,
            check,
            5, 6, 6, seeds
        )
        cp.z += 9