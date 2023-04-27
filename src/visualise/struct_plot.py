from src.logger import StructLogger
import matplotlib.pyplot as plt 
import os
import numpy as np
import math
from typing import List

class StructPlot(StructLogger):
    def __init__(self, out="out/struct"):       
        # Make output files
        self.out = out
        try:
            os.makedirs(out)
        except FileExistsError:
            pass


    def __plot_average(self, ax, data, title):
        # format data
        avg_data = [np.average(x) for x in data]
        ax.plot(
            range((len(data))),
            avg_data
        )
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Avg structure value")

    def plot(self, titles, loggers:List[StructLogger]):
        assert type(loggers) == list
        assert type(loggers[0]) == StructLogger
        assert len(titles) == len(loggers)
        # Get number of subplots
        dim = math.ceil(math.sqrt(len(loggers)))
        
        if dim == 1:
            fix, ax = plt.subplots(figsize=(10,10))
        else:    
            # Make and save graph
            fig, ax = plt.subplots(dim, dim, figsize=(10, 10))
            
        if dim == 1:
            self.__plot_average(ax, loggers[0].get_scores(), titles[0])
        else:
            # Fill graph
            idx = 0
            for i in range(dim):
                for j in range(dim): 
                    self.__plot_average(ax[i][j], loggers[i].get_scores(), titles[i])
                    idx += 1
            
        # Save to file
        plt.tight_layout()
        plt.savefig(self.out)