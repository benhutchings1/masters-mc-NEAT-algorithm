from src.House_Fitness import novelty
from src import neat

def test_distance():
    nov = novelty.Novelty()
    checkpoint_genomes = neat.load_genomes_checkpoint("tests/testdata/NEAT-checkpoint")
    
    for __, genome in checkpoint_genomes:
        assert nov.distance(genome, genome, 1, 1) == 0.0
    print("Distance test passed")
    
