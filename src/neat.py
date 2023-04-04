import neat
import numpy as np

def run(config_file, fitness_function, checkpoint_path="checkpoints/",
        n_generations=100):
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

def config_inputs(path, value):
    # read file
    with open(path, "r") as fs:
        lines = fs.readlines()
    # edit line
    for l in range(len(lines)):
        if lines[l][:10] == "num_inputs":
            lines[l] = f"num_inputs              = {value}\n"
    
    # write back lines
    with open(path, "w") as fs:
        fs.writelines(lines)
    
