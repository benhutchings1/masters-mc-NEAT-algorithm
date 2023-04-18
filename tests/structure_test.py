from src.House_Fitness import fitness_functions
from src.Blocks import block_interactions
import numpy as np

def test_structure():
    house = "tests/testdata/house.csv"
    bi = block_interactions.BlockInterface(block_path="src/Blocks/blocks.csv", connect=False)
    house = bi.read_np(house)
    print("House Fitness")
    for key, value in fitness_functions.single_structure_fitness(input=[0, 0, 0, 13, 23, 23], output=house).items():
        print(f"{key}: {value}")
    print()
    for key, value in fitness_functions.single_structure_fitness(input=[0, 0, 0, 13, 23, 23], output=np.random.rand(8, 8, 9)).items():
        print(f"{key}: {value}")
    