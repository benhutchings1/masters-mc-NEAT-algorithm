import numpy as np
import random 
from src.Blocks import block_interactions
import time

class HouseGenerator(block_interactions.BlockInterface):
    def __init__(self, blockspace=None, block_path=None, sample_size=3):
        super().__init__(block_path, connect=True)
        if blockspace is None:
            self.blockspace = range(len(self.blocklist))
        else:   
            self.blockspace = blockspace    
        self.sample_size = 3
        
    def generate(self, height, length, width):
        out = np.zeros((height, (2*length) + (2*width) - 4)).astype(int)
        time.sleep(0.1)
        return self.__gen_vert_symmetry(out, length, width, failure_rate=0.1)
        
    def __gen_vert_symmetry(self, arr, length, width, failure_rate):
        length_mid = self.__get_midpoint_bound(length)
        # Get both sides across symmetry line    
        boundry = length_mid[1]+2*(length_mid[0]+1)+(width-2) 
        sample_size = round(np.random.normal(self.sample_size, 0.25, 1)[0])
        sample = random.sample(self.blockspace, sample_size)
        for i in range(arr.shape[0]):
            # Get both sides of vert
            a_idx = list(range(length_mid[1], boundry))
            b_idx = list(reversed(range(length_mid[0]+1))) + list(reversed(range(len(arr[i]) - ((width-2)+length_mid[0]+1), len(arr[i]))))
            # Get between boundry
            other = []
            for j in range(arr.shape[1]):
                if not j in a_idx and not j in b_idx:
                    other.append(j)
            # Fill one side 
            a = random.choices(sample, k=len(a_idx))
            b = a
            # Fill sides
            for idx, val in zip(a_idx + b_idx, a + b):
                arr[i][idx] = val
            # Fill boundry
            t = random.sample(sample, k=1)
            for idx in other:
                arr[i][idx] = t[0]
        return arr
        
        
    def __get_midpoint_bound(self, x):
        x = x-2
        if x%2 == 0:
            return (int(x/2), int(x/2)+1)
        else:
            return (int(((x+1)/2)-1), int(((x+1)/2)+1))