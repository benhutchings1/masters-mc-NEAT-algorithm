import neat
import numpy as np

def run(config_file:str, fitness_function:object, checkpoint_path="checkpoints/",
        n_generations=100):
    """
    Performs NEAT model creation and evolution for n_generations 
    Configuration for NEAT given by config file 
    Models are incrementally checkpointed and saved in checkpoint_path
    returns the best performing model
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_path + "/NEAT-checkpoint-"))

    # Run for up to n generations.
    winner = p.run(fitness_function, n_generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    ## Use winning model to build neighbourhood
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    return winner_net

def edit_config(path:str, input_size=None, pop_size=None,
                       fitness_threshold=None) -> None:
    """
    Updates NEAT configuration file with input size given by value
    """
    # read file
    with open(path, "r") as fs:
        lines = fs.readlines()
    # edit line
    for l in range(len(lines)):
        if not input_size is None:
            if lines[l][:10] == "num_inputs":
                lines[l] = f"num_inputs              = {input_size}\n"
        
        if not pop_size is None:
            if lines[l][:8] == "pop_size":
                lines[l] = f"pop_size              = {pop_size}\n"
    
        if not fitness_threshold is None:
            if lines[l][:17] == "fitness_threshold":
                lines[l] = f"fitness_threshold              = {fitness_threshold}\n"

    # write back lines
    with open(path, "w") as fs:
        fs.writelines(lines)
    
