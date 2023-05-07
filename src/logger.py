import os
import csv
from neat.reporting import BaseReporter, StdOutReporter
from src.house_fitness import structure_functions as house_func
from src.roof_fitness import structure_functions as roof_func
import time
import itertools
import numpy as np
from skimage import morphology as morph

class Logger():
    def __init__(self, log_path, filename, overwrite=True, header=None):
        self.path = log_path
        self.filepath = os.path.join(log_path, filename)
        # Either overwrite or check file exists
        if overwrite:
            self.__make_directory()
            self.__make_file()
        else:
            if not os.path.isfile(self.filepath):
                raise FileNotFoundError(self.filepath)
        
        self.first_time = overwrite
        
        # optional initialise file with headers
        if header is not None:
            self.add_header(header)
    
    def __make_directory(self):
        # Make checkpoint path if non-existant
        try:
            os.makedirs(self.path)
        except FileExistsError:
            pass
        # Format directory
        if self.path[-1] != "/":
            self.path += "/"

        return self.path

    def __make_file(self):
        with open(self.filepath, "w+"):
            pass
   
    def log_value(self, val):
        with open(self.filepath, "a+") as fs:
            wr = csv.writer(fs, delimiter=",")
            wr.writerow(val)
    
    def log_values(self, values:list):
        assert type(values) == list
        
        with open(self.filepath, "a+") as fs:
            for val in values:
                wr = csv.writer(fs, delimiter=",")
                wr.writerow(val)
    
    def add_header(self, header):
        # Remove file and create
        os.remove(self.filepath)
        self.__make_file()
        # Write header
        self.log_value(header)

            

class NoveltyLogger(Logger):
    def __init__(self, log_path, filename, header=None, overwrite_log=True):
        super().__init__(log_path, header=header, filename=filename, overwrite=overwrite_log)

    def log_iteration(self, values):
        self.log_value(["##Gen##"])
        self.log_values(values)
        
    def read_file(self):
        pass


class StructLogger(Logger):
    def __init__(self, log_path, filename, header=None, overwrite_log=True):
        super().__init__(log_path, header=header, filename=filename, overwrite=overwrite_log)
    
    def start_gen(self):
        self.log_value(["##Gen##"])
    
    def read_file(self, return_headers=False):
        data = []
        with open(self.filepath, "r") as fs:
            # Open file
            rr = csv.reader(fs, delimiter=",")
            # Skip header
            headers = next(rr, None)
            # Read generation data
            buff = []
            for line in rr:
                if line == ["##Gen##"]:
                    data.append(np.array(buff))
                    buff = []
                else:
                    buff.append(line)
        
        if return_headers:
            return (headers, data[1:])
        else:
            return data[1:]
                
    def get_scores(self):
        data = self.read_file()
        for i in range(len(data)):
            data[i] = data[i].astype(float)
            
        scores = []
        for gen in data:
            scores.append(np.array([np.average(x) for x in gen]))
        
        return scores
        

class StatsLogger(Logger, StdOutReporter):
    def __init__(self, log_path, filename, header=None, overwrite_log=True):
        Logger.__init__(self, log_path, header=header, filename=filename, overwrite=overwrite_log)
        StdOutReporter.__init__(self, True)
        
        if overwrite_log:
            self.add_header(["generation", "generation_time"])
    
    def end_generation(self, config, population, species_set):
        self.log_value([self.generation, time.time() - self.generation_start_time])
        return super().end_generation(config, population, species_set)

class MultiStructureLogger(Logger):
    def __init__(self, log_path, filename, overwrite_log=True, header=None):
        super().__init__(log_path, filename, overwrite_log, header)
        self.constructions = []
        self.block_counts = {}

    def add_construction(self, house, roof, house_input, roof_input):
        # Store construction and structure score
        con = Construction(house, roof, house_input[1], house_input[2]) 
        con.roof_struct_score = roof_func.single_structure_score(roof_input, roof, avg=True)
        con.house_struct_score = house_func.single_structure_score(house_input, house, avg=True)
        self.constructions.append(con)
        
        # Record block information when entering
        counts = np.unique(house, return_counts=True)
        # Add frequencies to dictionary
        for id, count in zip(counts[0], counts[1]):
            if not id in self.block_counts:
                self.block_counts[id] = 0
            self.block_counts[id] += count
    
    def get_building_variance(self):
        ids = {}
        house_vars = []
        roof_vars = []
        # Get all unique combinations
        for con in self.constructions:
            if not (con.length, con.width) in ids:
                ids[(con.length, con.width)] = []
            ids[(con.length, con.width)].append(con)
        
        for (l, w), cons in ids.items():
            for a, b in itertools.permutations(cons, 2):
                h,r  = self.construction_diff(a, b)
                house_vars.append(h)
                roof_vars.append(r)
        return np.median(house_vars), np.median(roof_vars)
        
    def construction_diff(self, a:np.array, b:np.array):
        assert a.length == b.length
        assert a.width == b.width
        a = morph.label(a, connectivity=2)
        b = morph.label(b, connectivity=2)
        raise NotImplementedError() ###TEST THIS
        house_var = 1 - (np.count_nonzero(a.house==b.house) / (a.house.shape[0] * a.house.shape[1]))
        roof_var = 1 - (np.count_nonzero(a.roof==b.roof) / (a.roof.shape[0] * a.roof.shape[1]))
        
        return (house_var, roof_var)
        
class Construction:
    def __init__(self, house, roof, length, width) -> None:
        self.house = house
        self.roof = roof
        self.length = length
        self.width = width
        self.roof_struct_score = 0
        self.house_struct_score = 0
        