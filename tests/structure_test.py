from src.Fitness import fitness_functions
from src.Blocks import block_interactions

def test_structure():
    house = "tests/testdata/house.csv"
    bi = block_interactions.BlockReader(block_path="blocks.csv", connect=False)
    house = bi.read_np(house)
    print(fitness_functions.single_structure_fitness(input=[0, 0, 0, 13, 23, 23], output=house))
    