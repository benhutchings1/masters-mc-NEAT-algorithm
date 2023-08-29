import neat
import os
from src.Blocks.block_interactions import BlockInterface
from src.logger import StatsLogger
import shutil

class Neat(BlockInterface):
    def __init__(self, config_file:str, block_path:str=None, stats_log_path:str=None, fitness_func:object=None, checkpoint_path:str=None,
            n_generations=1000, n_input=None, n_pop=None, n_output=None, overwrite=False, checkpoint_rate=10):       
        # Run parent init
        if not block_path is None:
            super().__init__(block_path=block_path, connect=False)

        # Setup new checkpoint if overwrite is True
        if overwrite:
            if os.path.exists(checkpoint_path):
                # Remove previous checkpoints
                shutil.rmtree(checkpoint_path)

        # Make checkpoint path if non-existant
        if overwrite:
            try:
                os.makedirs(checkpoint_path)
            except FileExistsError:
                pass
        else:
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint path {checkpoint_path} doesn't exist")
        self.checkpoint:str = checkpoint_path
        
        # Make config file from config path
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
        self.config_path:str = config_file     
          
        # Edit config file and save configs
        self.n_input:int = n_input
        self.pop_size:int = n_pop
        self.n_output:int = n_output
        self.n_gen:int = n_generations
        self.edit_config()
        self.checkpoint_rate = checkpoint_rate
        
        # Save fitness function
        self.fitness = fitness_func
        
        # Make stats logger
        if not stats_log_path is None: 
            self.stats_logger = StatsLogger(stats_log_path, "stats.csv", overwrite_log=overwrite)
    
    def run(self, population:neat.Population=None, checkpoint_prefix="NEAT-checkpoint-"):
        """
        Performs NEAT model creation and evolution for n_generations 
        Configuration for NEAT given by config file 
        Models are incrementally checkpointed and saved in checkpoint_path
        returns the best performing model
        """

        # Create the population, which is the top-level object for a NEAT run.
        if population is None:
            p = neat.Population(self.config)
        else:
            # Use population from checkpoint
            p = population
        
        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(self.stats_logger)
        p.add_reporter(neat.Checkpointer(self.checkpoint_rate, filename_prefix=self.checkpoint + checkpoint_prefix))

        # Run evolution, save models to checkpoint
        p.run(self.fitness, self.n_gen)


    def edit_config(self) -> None:
        """
        Updates NEAT configuration file with input size given by value
        """
        # read file
        with open(self.config_path, "r") as fs:
            lines = fs.readlines()
        
        # edit line
        for l in range(len(lines)):
            if not self.n_input is None:
                if lines[l][:10] == "num_inputs":
                    lines[l] = f"num_inputs              = {self.n_input}\n"
            
            if not self.pop_size is None:
                if lines[l][:8] == "pop_size":
                    lines[l] = f"pop_size              = {self.pop_size}\n"
                                
            if not self.n_output is None:
                if lines[l][:11] == "num_outputs":
                    lines[l] = f"num_outputs             = {self.n_output}\n"

           # write back lines
        with open(self.config_path, "w") as fs:
            fs.writelines(lines)
        
    def load_checkpoint(self, checkpoint_name:str) -> neat.Population:
        return neat.Checkpointer().restore_checkpoint(os.path.join(self.checkpoint ,checkpoint_name))

    def run_from_checkpoint(self, checkpoint_name, checkpoint_prefix):
        return self.run(self.load_checkpoint(checkpoint_name, checkpoint_prefix))

    def load_genomes_checkpoint(self, checkpoint_name:str):
        return self.load_checkpoint(checkpoint_name).population.items()