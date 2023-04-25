from src.roof_fitness import fitness_functions as roof_fit_fn
from src.house_fitness import fitness_functions as house_fit_fn
from src import neat as nt, novelty
import numpy as np
import unittest
from src.Blocks import block_interactions

class TestRoofFitness(unittest.TestCase):
    def test_complexity(self):
        a = np.array([
            [1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1]
        ])
        b = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        self.assertGreater(roof_fit_fn.fit_complexity(None, a), roof_fit_fn.fit_complexity(None, b))
        print("Complexity test passed")
        
    def test_vert_symmetry(self):
        a = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        b = np.array([
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1]
        ])
        
        c = np.array([
            [1, 3, 0, 1, 2],
            [1, 3, 0, 1, 2],
            [1, 3, 0, 1, 2],
            [1, 3, 0, 1, 2],
            [1, 3, 0, 1, 2]
        ])
        
        self.assertTrue(roof_fit_fn.fit_vert_symmetry(a) == 1.0)
        self.assertTrue(roof_fit_fn.fit_vert_symmetry(b) == 1.0)
        self.assertTrue(roof_fit_fn.fit_vert_symmetry(c) == 0.0)
        print("Vertical symmetry passed")
        

    def test_horiz_symmetry(self):
        a = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        b = np.array([
            [3, 1, 1, 3, 3],
            [1, 2, 2, 3, 1],
            [3, 4, 5, 6, 7],
            [1, 2, 2, 3, 1],
            [3, 1, 1, 3, 3]
        ])
        
        c = np.array([
            [1, 2, 4, 4, 5],
            [2, 2, 2, 3, 3],
            [10, 30, 0, 1, 0],
            [3, 3, 3, 2, 2],
            [5, 4, 3, 2, 1]
        ])
        self.assertTrue(roof_fit_fn.fit_horiz_symmetry(a) == 1.0)  
        self.assertTrue(roof_fit_fn.fit_horiz_symmetry(b) == 1.0)
        self.assertTrue(roof_fit_fn.fit_horiz_symmetry(c) == 0.0)
        print("Horizontal symmetry passed")

    def test_compliance(self):
        a = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        b = np.array([
            [-1, 1, 1, 3, 3],
            [1, 2, 2, 3, 1],
            [3, 4, 5, 6, 7],
            [1, 2, 2, 3, 1],
            [3, 1, 1, 3, 3]
        ])
        
        c = np.array([
            [1, 2, 3, 4, 5],
            [2, 2, 2, 3, 3],
            [10, 30, 0, 1, 0],
            [3, 3, 3, 2, 2],
            [5, 4, 3, 2, 1]
        ])
        
        self.assertTrue(roof_fit_fn.fit_compliance([2, 1, 1], a))
        self.assertFalse(roof_fit_fn.fit_compliance([10, 1, 1], b))
        self.assertFalse(roof_fit_fn.fit_compliance([5, 1, 1], c))
        print("Compliance test passed")

class TestStructureFitness(unittest.TestCase):        
    def __init__(self):
        self.bi = block_interactions.BlockInterface(connect=False)    
    def test_bounding_wall(self):
        data = self.get_data("bounding_wall")
        self.assertTrue(house_fit_fn.fit_bounding_wall(None, [], data[0]) == 1.0)
        self.assertTrue(house_fit_fn.fit_bounding_wall(None, [], data[1]) == 0.0)
        self.assertTrue(house_fit_fn.fit_bounding_wall(None, [], data[2]) == 0.5)
        print("Bounding wall passed")
        
    def test_door(self):
        data = self.get_data("door")
        self.assertTrue(house_fit_fn.fit_door(None, None, data[0]) == 1.0)
        self.assertTrue(house_fit_fn.fit_door(None, None, data[1]) == 0.0)
        self.assertTrue(house_fit_fn.fit_door(None, None, data[2]) == 0.0)
        print("Door test passed")
    
    def test_airspace(self):
        data = self.get_data("airspace")
        self.assertTrue(house_fit_fn.fit_airspace(None, None, data[0]) == 1.0)
        self.assertTrue(house_fit_fn.fit_airspace(None, None, data[1]) == 0.0)
        self.assertTrue(house_fit_fn.fit_airspace(None, None, data[2]) == 0.5)
        print("Airspace test passed")
    
    def test_seed_blocks(self):
        data = self.get_data("seed_blocks")
        self.assertTrue(house_fit_fn.fit_seed_blocks(None, [3,5,5,1,2,3], data[0]) == 1.0)
        self.assertTrue(house_fit_fn.fit_seed_blocks(None, [3,5,5,1,2,3], data[1]) == 0.0)
        self.assertTrue(house_fit_fn.fit_seed_blocks(None, [3,5,5,1,2,3], data[2]) == (1/3))
        print("Seed block test passed")
    
    def test_vert_symmetry(self):
        data = self.get_data("vert_symmetry")
        self.assertTrue(house_fit_fn.fit_vert_symmetry(data[0]) == 1.0)
        self.assertTrue(house_fit_fn.fit_vert_symmetry(data[1]) == 0.0)
        self.assertTrue(house_fit_fn.fit_vert_symmetry(data[2]) == 0.5)
        print("Vertical symmetry passed")
    
    def test_horiz_symmetry(self):
        data = self.get_data("horiz_symmetry")
        self.assertTrue(house_fit_fn.fit_horiz_symmetry(data[0]) == 1.0)
        self.assertTrue(house_fit_fn.fit_horiz_symmetry(data[1]) == 0.0)
        self.assertTrue(house_fit_fn.fit_horiz_symmetry(data[2]) == 0.5)
        print("Horizonal symmetry passed")
           
    def get_data(self, folder):
        out = []
        for letter in ["a", "b", "c"]:
            out.append(self.bi.read_np(f"testdata/structure_data/{folder}/{letter}.txt").astype(int))
        return out
        

# class TestNoveltyFitness(unittest.TestCase):
#     def test_distance():
#         nov = novelty.Novelty()
#         checkpoint_genomes = nt.load_genomes_checkpoint("tests/testdata/NEAT-checkpoint")
        
#         for __, genome in checkpoint_genomes:
#             assert nov.distance(genome, genome, 1, 1) == 0.0
#         print("Distance test passed")


if __name__ == "__main__":
    roof = TestRoofFitness()
    house = TestStructureFitness()
    roof.test_complexity()
    roof.test_compliance()
    roof.test_horiz_symmetry()
    roof.test_vert_symmetry()
    print("\033[1;32;40mRoof tests passed\033[0;37m")
    
    house.test_vert_symmetry()
    house.test_seed_blocks()
    house.test_airspace()
    house.test_door()
    house.test_horiz_symmetry()
    house.test_bounding_wall()
    print("\033[1;32;40mStructure tests passed\033[0;37m")