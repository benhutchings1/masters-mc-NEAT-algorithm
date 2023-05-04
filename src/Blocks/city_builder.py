from . import block_interactions
from src.house_fitness import generate_building
from src.roof_fitness import generate_roof
from src import neat
import random

class CityBuilder(block_interactions.BlockInterface):
    def __init__(self, block_path=None, mutate_chance=0.1):
        super().__init__(block_path, connect=False)
        self.block_path = block_path
        self.house_pop = []
        self.roof_pop = []
        self.WIDTH = 6
        self.LENGTH = 6
        self.HOUSE_HEIGHT = 5
        self.ROOF_HEIGHT = 4
        self.seeds = [10, 20, 30]
        self.mutate_chance = mutate_chance
        
        
        
    def read_in_pop(self, config_path, checkpoint_path, checkpoint_name, house=True):
        nt = neat.Neat(config_file=config_path, checkpoint_path=checkpoint_path, overwrite=False)
        cp = nt.load_checkpoint(checkpoint_name)
        for i in list(cp.population.items()):
            if house:
                self.house_pop.append(i[1])
            else:
                self.roof_pop.append(i[1])
    
    def place_city(self, x0=0, y0=-60, z0=0, orientation="N", gap=5, plot_size=20, iterations=10):
        if len(self.house_pop) == 0:
            raise ValueError("House populations cannot be empty")    

        if len(self.roof_pop) == 0:
            raise ValueError("Roof population cannot be empty")
        # Initial building specs
        
        
        # Initial position
        pos = {"x":x0, "z":z0}
        # Get direction to move in
        first_dir, second_dir = self.__conv_orientation(orientation)
        
        for it in range(iterations):
            # Start of diagonal movement
            temp_position = pos.copy()
            # Iterate down diagonal
            for __ in range(it+1):
                # Place at temp position
                # Random chance to mutate
                
                # Generate roof and house
                
                
                # Update temp position
                temp_position[first_dir[0]] -= first_dir[1] * plot_size
                temp_position[second_dir[0]] += second_dir[1] * plot_size
            # After all placed update position
            pos[first_dir[0]] += first_dir[1]*plot_size                 
    
    def __conv_orientation(self, orientation):
        if orientation == "N":
            return [["z", 1],["x", 1]]
        elif orientation == "E":
            return [["x", 1],["z", -1]]
        elif orientation == "S":
            return [["z", -1],["x", -1]]
        elif orientation == "W":
            return [["x", -1],["z", 1]]

    def __mutate(self):
        values = [self.WIDTH, self.LENGTH, self.HOUSE_HEIGHT, self.ROOF_HEIGHT]
        lims = [[5, 15],[5, 15],[2, 7],[2, 5]]
        chances = random.sample(range(0, 100), 5)
        
        for val, chan, lim in zip(values, chances, lims):
            if chan < 100 * (self.mutate_chance/2):
                if val - 1 >= lims[0]:
                    val -= 1
            
            if chan > 100 - (100 * (self.mutate_chance/2)):
                if val + 1 <= lims[1]:
                    val += 1
        
        ## DO SEED MUTATE AND TEST THIS
    
    
