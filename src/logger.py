import os
import csv
from neat.reporting import BaseReporter, StdOutReporter
import time

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
    
    def read_file(self):
        pass


class StatsLogger(Logger, StdOutReporter):
    def __init__(self, log_path, filename, header=None, overwrite_log=True):
        Logger.__init__(self, log_path, header=header, filename=filename, overwrite=overwrite_log)
        StdOutReporter.__init__(self, True)
        
        if overwrite_log:
            self.add_header(["generation", "generation_time"])
    
    def end_generation(self, config, population, species_set):
        self.log_value([self.generation, time.time() - self.generation_start_time])
        return super().end_generation(config, population, species_set)