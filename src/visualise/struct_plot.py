from src.logger import StructLogger
import matplotlib.pyplot as plt 
import os
import numpy as np
import math
from typing import List
from scipy.ndimage import uniform_filter1d

class StructPlot(StructLogger):
    def __init__(self, out="out/struct"):       
        # Make output files
        if out is not None:
            self.out = out
            try:
                os.makedirs(out)
            except FileExistsError:
                pass

    def __plot_values(self, ax, data, title, const=0):
        # format data
        avg_data = self.const_alt(data, np.average, const)
        max_data = self.const_alt(data, np.max, const)
        min_data = self.const_alt(data, np.min, const)
        ax.plot(
            range((len(data))),
            max_data,
            label="Max Structure Score",
            alpha=0.75
        )
        ax.plot(
            range((len(data))),
            min_data,
            label="Min Structure Score",
            alpha=0.75
        )
        ax.plot(
            range((len(data))),
            avg_data,
            label="Avg Struct Score",
            color="black"
        )
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Structure Score")
        ax.legend()

    def const_alt(self, data, operation, const):
        out = []
        for x in data:
            y = operation(x) - const
            if y < 0:
                out.append(0)
            elif y > 1:
                out.append(1)
            else:
                out.append(y)
        return out
            
    
    def plot(self, titles, loggers:List[StructLogger], super_plot_title, filename, const=0, max_read=None):
        assert type(loggers) == list
        assert type(loggers[0]) == StructLogger
        assert len(titles) == len(loggers)
        # Get number of subplots        
        if len(loggers) == 1:
            fix, ax = plt.subplots(figsize=(10,10))
        else:    
            # Make and save graph
            fig, ax = plt.subplots(1, len(loggers), figsize=(10, 10))

        if len(loggers) == 1:
            self.__plot_values(ax[0], loggers[0].get_scores(max_read=max_read), titles[0])
        else:
            # Fill graph
            for i in range(len(loggers)):
                self.__plot_values(ax[0][i], loggers[i].get_scores(max_read=max_read[i]), titles[i], const=const[i])
                
            
        # Save to file
        plt.tight_layout()
        plt.suptitle(super_plot_title)
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(self.out, filename))
        
    def individual_scores_plot(self, logger:StructLogger, plot_title, filename):
        headers, data = logger.read_file(return_headers=True)
        fig, ax = plt.subplots(len(headers), figsize=(10, len(headers)*5))
        
        # Gather data
        format_data = {head:[] for head in headers}
        
        for gen in data:
            gen = gen.astype(float)
            for i, header in enumerate(headers):
                format_data[header].append(np.average(gen[:, i]))
        
        for axi, header in zip(ax, headers):
            axi.plot(
                range(len(format_data[header])),
                format_data[header]
            )
            axi.set_title(f"{header} plot")
        
        # Save to file
        plt.tight_layout()
        plt.suptitle(plot_title)
        plt.savefig(os.path.join(self.out, filename))
        
    def highest_moving_average(self, logger:StructLogger, window_size):
        data = [np.average(x.astype(float)) for x in logger.read_file()]
        return np.max(uniform_filter1d(data, window_size))
