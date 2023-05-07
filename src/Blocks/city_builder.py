from . import block_interactions
from src.house_fitness import generate_building
from src.roof_fitness import generate_roof
from src import neat as my_neat
from src.logger import MultiStructureLogger
import neat
import random

class CityBuilder(block_interactions.BlockInterface):
    def __init__(self, block_path, house_config, roof_config, log_path, filename, overwrite_log, mutate_chance=0.1, connect=True):
        super().__init__(block_path, connect=connect)
        self.block_path = block_path
        self.house_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            house_config)
        self.roof_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            roof_config)
        self.house_pop = []
        self.roof_pop = []
        self.WIDTH = 6
        self.LENGTH = 6
        self.HOUSE_HEIGHT = 5
        self.ROOF_HEIGHT = 4
        self.seeds = [10, 20, 30]
        self.mutate_chance = mutate_chance
        self.logger = MultiStructureLogger(log_path=log_path, filename=filename, overwrite_log=overwrite_log)
        
        
        
    def read_in_pop(self, config_path, checkpoint_path, checkpoint_name, house=True):
        nt = my_neat.Neat(config_file=config_path, checkpoint_path=checkpoint_path, overwrite=False)
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
        
        # Initial position
        pos = {"x0":x0, "z0":z0}
        # Get direction to move in
        first_dir, second_dir = self.__conv_orientation(orientation)
        
        for it in range(iterations):
            # Start of diagonal movement
            temp_position = pos.copy()
            # Iterate down diagonal
            for __ in range(it+1):
                # Place at temp position
                # Random chance to mutate
                self.mutate_params()
                # Choose random models
                house_model = neat.nn.FeedForwardNetwork.create(random.choice(self.house_pop), self.house_config)
                roof_model = neat.nn.FeedForwardNetwork.create(random.choice(self.roof_pop), self.roof_config)                                
                # Generate roof and house
                house_input = [self.HOUSE_HEIGHT, self.LENGTH, self.WIDTH]+self.seeds
                roof_input = [self.ROOF_HEIGHT, self.LENGTH, self.WIDTH]
                house = generate_building.generate(0, house_model, house_input)[1]
                roof = generate_roof.generate(0, roof_model, roof_input)[1]
                
                # Place generated building
                if self.to_connect:
                    self.place_house(house, self.WIDTH, self.LENGTH, y0=y0, orientation=orientation,
                                    isblocklist=True, **temp_position)
                    self.place_roof(roof, orientation=orientation, y0=y0+self.HOUSE_HEIGHT, **temp_position)
                
                # Log building
                self.logger.add_construction(house, roof, house_input, roof_input)
                
                # Update temp position
                temp_position[first_dir[0]] -= (first_dir[1] * plot_size) + gap
                temp_position[second_dir[0]] += (second_dir[1] * plot_size) + gap
            
            # After all placed update position
            pos[first_dir[0]] += first_dir[1]*plot_size                 
    
    def __conv_orientation(self, orientation):
        if orientation == "N":
            return [["z0", -1],["x0", 1]]
        elif orientation == "E":
            return [["x0", 1],["z0", 1]]
        elif orientation == "S":
            return [["z0", 1],["x0", -1]]
        elif orientation == "W":
            return [["x0", -1],["z0", -1]]

    def mutate_params(self):
        self.WIDTH = self.mutate(self.WIDTH, [5, 10])
        self.LENGTH = self.mutate(self.LENGTH, [5, 10])
        self.HOUSE_HEIGHT = self.mutate(self.HOUSE_HEIGHT, [3, 7])
        self.ROOF_HEIGHT = self.mutate(self.ROOF_HEIGHT, [2, 5])
        self.seeds[0] = self.mutate(self.seeds[0], [1, len(self.blocklist) - 1], add=False) 
        self.seeds[1] = self.mutate(self.seeds[1], [1, len(self.blocklist) - 1], add=False)
        self.seeds[2] = self.mutate(self.seeds[2], [1, len(self.blocklist) - 1], add=False)
    
    def mutate(self, value, lim, add=True):
        rnd = random.randint(0, 100)
        
        if rnd <= (100 * self.mutate_chance):
            if add:
                if not value - 1 < lim[0]:
                    value -= 1
            else:
                value = random.randint(lim[0], lim[1])
                
        elif rnd >= 100 - (100 * self.mutate_chance):
            if add:
                if not value + 1 > lim[1]:
                    value += 1
            else:
                value = random.randint(lim[0], lim[1])
        
        return value