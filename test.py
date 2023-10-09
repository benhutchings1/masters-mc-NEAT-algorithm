from src.roof_fitness import structure_functions as roof_struct_fn
from src.house_fitness import structure_functions as house_struct_fn
import numpy as np
import unittest
from src.Blocks import block_interactions
import CONFIG

class TestRoofFitness(unittest.TestCase):
    # Test cases to ensure roof fitness functions are correct
    def test_complexity(self):
        # Testing data
        # a: high complexity roof
        # b: low complexity roof
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
        self.assertGreater(roof_struct_fn.score_complexity(None, a), roof_struct_fn.score_complexity(None, b))
        print("Complexity test passed")
        
    def test_vert_symmetry(self):
        # Testing data
        # a: full vertical symmetry
        # b: full vertical symmetry
        # c: no vertical symmetry
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
        
        self.assertTrue(roof_struct_fn.score_vert_symmetry(a) == 1.0)
        self.assertTrue(roof_struct_fn.score_vert_symmetry(b) == 1.0)
        self.assertTrue(roof_struct_fn.score_vert_symmetry(c) == 0.0)
        print("Vertical symmetry passed")
        

    def test_horiz_symmetry(self):
        # Testing data
        # a: full horizontal symmetry
        # b: full horizontal symmetry
        # c: no horizontal symmetry
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
        self.assertTrue(roof_struct_fn.score_horiz_symmetry(a) == 1.0)  
        self.assertTrue(roof_struct_fn.score_horiz_symmetry(b) == 1.0)
        self.assertTrue(roof_struct_fn.score_horiz_symmetry(c) == 0.0)
        print("Horizontal symmetry passed")

    def test_compliance(self):
        # Testing data
        # a: a compliant roof 
        # b: a compliant roof
        # c: a non compliant roof
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
        
        self.assertTrue(roof_struct_fn.score_compliance([2, 1, 1], a))
        self.assertTrue(roof_struct_fn.score_compliance([10, 1, 1], b))
        self.assertFalse(roof_struct_fn.score_compliance([5, 1, 1], c))
        print("Compliance test passed")

class TestStructureFitness(unittest.TestCase):      
    # Test structure fitness functions
    # Test data path can be found in CONFIG.py
    def __init__(self):
        self.bi = block_interactions.BlockInterface(CONFIG.BLOCK_PATH, connect=False)    
        
    def test_bounding_wall(self):
        data = self.get_data("bounding_wall")
        self.assertTrue(house_struct_fn.score_bounding_wall(None, [], data[0]) == 0.0)
        self.assertTrue(house_struct_fn.score_bounding_wall(None, [], data[1]) == 0.5)
        self.assertTrue(house_struct_fn.score_bounding_wall(None, [], data[2]) == 1.0)
        print("Bounding wall passed")
        
    def test_door(self):
        data = self.get_data("door")
        door_id = house_struct_fn.get_door_id()
        self.assertFalse(house_struct_fn.score_door(None, None, data[0]) == 1.0)
        self.assertTrue(house_struct_fn.score_door(None, None, data[1]) == 0.0)
        self.assertTrue(house_struct_fn.score_door(None, None, data[2]) == 0.0)
        print("Door test passed")
    
    def test_seed_blocks(self):
        data = self.get_data("seed_blocks")
        self.assertTrue(house_struct_fn.score_seed_blocks(None, [4,4,4,1,2,3], data[0]) == 1.0)
        self.assertTrue(house_struct_fn.score_seed_blocks(None, [4,4,4,1,2,3], data[1]) == 0.0)
        self.assertTrue(house_struct_fn.score_seed_blocks(None, [4,4,4,1,2,3], data[2]) == (1/3))
        print("Seed block test passed")
    
    def test_vert_symmetry(self):
        data = self.get_data("vert_symmetry")
        self.assertTrue(house_struct_fn.score_vert_symmetry([4,4,4,1,2,3], data[0]) == 1.0)
        self.assertTrue(house_struct_fn.score_vert_symmetry([4,4,4,1,2,3], data[1]) == 0.0)
        self.assertTrue(house_struct_fn.score_vert_symmetry([4,4,4,1,2,3], data[2]) == 0.5)
        print("Vertical symmetry passed")
    
    def test_horiz_symmetry(self):
        data = self.get_data("horiz_symmetry")
        self.assertTrue(house_struct_fn.score_horiz_symmetry([4,4,4,1,2,3], data[0]) == 1.0)
        self.assertTrue(house_struct_fn.score_horiz_symmetry([4,4,4,1,2,3], data[1]) == 0.0)
        self.assertTrue(house_struct_fn.score_horiz_symmetry([4,4,4,1,2,3], data[2]) == 0.5)
        print("Horizonal symmetry passed")
    
    def test_full_structure(self):
        data = self.get_data("full_structure")
        data = [self.bi.convert_to_blocklist(d) for d in data]
        scores = []
        scores.append(house_struct_fn.single_structure_score([7, 7, 9, 5, 17, 64], data[0]))
        scores.append(house_struct_fn.single_structure_score([4, 5, 6, 5, 17, 64], data[1]))
        scores.append(house_struct_fn.single_structure_score([4, 5, 6, 5, 17, 64], data[2]))

        for key in scores[0].keys():
            self.assertGreaterEqual(scores[0][key], scores[1][key])
            self.assertGreaterEqual(scores[0][key], scores[2][key])
        print("Full test passed")
    
    def get_data(self, folder):
        out = []
        for letter in ["a", "b", "c"]:
            out.append(self.bi.read_np(f"{CONFIG.TEST_DATA_PATH}/structure_data/{folder}/{letter}.txt").astype(int))
        return out
    
if __name__ == "__main__":
    # Run all tests
    roof = TestRoofFitness()
    house = TestStructureFitness()
    roof.test_complexity()
    roof.test_compliance()
    roof.test_horiz_symmetry()
    roof.test_vert_symmetry()
    print("\033[1;32;40mRoof tests passed\033[0;37m")
    
    house.test_vert_symmetry()
    house.test_seed_blocks()
    house.test_door()
    house.test_horiz_symmetry()
    house.test_bounding_wall()
    house.test_full_structure()
    print("\033[1;32;40mStructure tests passed\033[0;37m")

    