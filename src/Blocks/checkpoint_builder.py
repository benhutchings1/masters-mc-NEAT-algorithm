from src.neat import Neat
from src.Blocks import block_interactions
import numpy as np
from src.roof_fitness import generate_roof
from src.house_fitness import generate_building
import neat

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
        
    def place_best_from_checkpoint(self, fitness, checkpoint_name, height, length, width, seeds=None):
        self.nt = Neat(self.config_path, self.block_path, self.checkpoint_path, "temp/",fitness, overwrite=False)
        self.nt.connect()
        model = self.nt.checkpoint_best_genome(checkpoint_name, as_model=True)

        if self.type == 1:
            self.roof_placer(model, height, length, width)
        else:
            if seeds is None or type(seeds) is not list:
                raise ValueError
            else:
                self.house_placer(model, height, length, width, seeds)
    
    def place_population_from_checkpoint(self, checkpoint_name, height, length, width, seeds=None):
        self.nt = Neat(self.config_path, self.block_path, self.checkpoint_path, "temp/", None, overwrite=False)
        self.nt.connect()
        population = self.nt.load_genomes_checkpoint(checkpoint_name)
        nets = [neat.nn.FeedForwardNetwork.create(gen, self.nt.config) 
                for __, gen in population
        ]
        if self.type == 1:
            builds = [
                self.roof_generator(net, height, width, length) for net in nets
            ]
        else:
            builds = [
                self.house_generator(net, height, width, length, seeds=seeds) for net in nets
            ]
        self.placer(builds, axis=0)
            
    def roof_generator(self, model, height, length, width):
        return self.nt.convert_heightmap(
                generate_roof.generate(0, model, [height, length, width])[1]
            )
        
    def house_generator(self, model, height, length, width, seeds):
        return generate_building.generate(0, model,
                                            [height, length, width]+seeds)[1]
        
    def placer(self, structures, axis=0):
        for struct in structures:
            self.nt.place_blocks_np(struct, x0=self.x, y0=self.y, z0=self.z,
                                isblocklist=True)
            if axis == 0:
                self.x += len(struct[0]) + 1
            else:
                self.z += len(struct)
     
if __name__ == "__main__":
    tp = 1
    if tp == 0:       
        config = "config/Roof-NEAT-config"
        checkpoints = "logs/house/roof/checkpoints"
        cp = CheckpointPlacer(0, -50, 0, 1,
                            config_path=config,
                            block_path="src/Blocks/blocks.csv",
                            checkpoint_path=checkpoints
                            )
        cp.place_population_from_checkpoint(
            checkpoint_name="NEAT-checkpoint-999",
            height=4, length=8, width=8
        )
    else:
        config = "config/House-NEAT-config"
        checkpoints = "data/dynamic/iter2/house/checkpoints/"
        cp = CheckpointPlacer(0, -60, 0, 2,
                            config_path=config,
                            block_path="src/Blocks/blocks.csv",
                            checkpoint_path=checkpoints
                            )
        cp.place_population_from_checkpoint(
            checkpoint_name="NEAT-checkpoint-99",
            height=8, length=8, width=8, seeds=[1, 2, 3]
        )