from src import neat
from src.house_fitness.fitness import Fitness as HouseFitness
from src.roof_fitness.fitness import Fitness as RoofFitness

class RoofInterface(neat.Neat):
    def __init__(self, config_file:str, block_path:str, log_root:str, n_generations=1000, 
                 n_input=None, n_pop=None, n_output=None, overwrite_logs=True, checkpoint_rate=10,
                 use_novelty=False, use_dynamic_novelty=False, novelty_ratio=None, squash_function=None):
        
        self.fit_func = RoofFitness(
            block_path,
            log_root+"novelty/",
            log_root+"struct/",
            overwrite=overwrite_logs,
            use_novelty=use_novelty,
            use_dynamic_novelty=use_dynamic_novelty,
            novelty_ratio=novelty_ratio,
            squash_function=squash_function
            )
        super().__init__(
            config_file=config_file, 
            block_path=block_path, 
            checkpoint_path=log_root+"checkpoints/", 
            stats_log_path=log_root+"stats/",
            overwrite=overwrite_logs,
            fitness_func=self.fit_func, 
            n_generations=n_generations, 
            n_input=n_input, 
            n_pop=n_pop, 
            n_output=n_output,
            checkpoint_rate=checkpoint_rate)

class HouseInterface(neat.Neat):
    def __init__(self, config_file:str, block_path:str, log_root:str, n_generations=1000, 
                 n_input=None, n_pop=None, n_output=None, overwrite_logs=True, checkpoint_rate=10,
                 use_novelty=False, use_dynamic_novelty=False, novelty_ratio=None, squash_function=None):
        
        self.fit_func = HouseFitness(
            block_path,
            log_root+"novelty/",
            log_root+"struct/",
            overwrite=overwrite_logs,
            use_novelty=use_novelty,
            use_dynamic_novelty=use_dynamic_novelty,
            novelty_ratio=novelty_ratio,
            squash_function=squash_function
            )
        super().__init__(
            config_file=config_file, 
            block_path=block_path, 
            checkpoint_path=log_root+"checkpoints/", 
            stats_log_path=log_root+"stats/",
            overwrite=overwrite_logs,
            fitness_func=self.fit_func, 
            n_generations=n_generations, 
            n_input=n_input, 
            n_pop=n_pop, 
            n_output=n_output,
            checkpoint_rate=checkpoint_rate)
        