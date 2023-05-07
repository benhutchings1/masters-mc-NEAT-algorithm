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

    def __plot_values(self, ax, data, title):
        # format data
        avg_data = [np.average(x) for x in data]
        max_data = [np.max(x) for x in data]
        min_data = [np.min(x) for x in data]
        ax.plot(
            range((len(data))),
            avg_data,
            label="Avg Struct Score"
        )
        ax.plot(
            range((len(data))),
            max_data,
            label="Max Structure Score"
        )
        ax.plot(
            range((len(data))),
            min_data,
            label="Min Structure Score"
        )
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Structure Score")
        ax.legend()

    def plot(self, titles, loggers:List[StructLogger], super_plot_title):
        assert type(loggers) == list
        assert type(loggers[0]) == StructLogger
        assert len(titles) == len(loggers)
        # Get number of subplots        
        if len(loggers) == 1:
            fix, ax = plt.subplots(figsize=(10,10))
        else:    
            # Make and save graph
            fig, ax = plt.subplots(len(loggers), figsize=(10, 10))
            
        if len(loggers) == 1:
            self.__plot_average(ax, loggers[0].get_scores(), titles[0])
        else:
            # Fill graph
            for i in range(len(loggers)):
                self.__plot_values(ax[i], loggers[i].get_scores(), titles[i])
                
            
        # Save to file
        plt.tight_layout()
        plt.suptitle(super_plot_title)
        plt.subplots_adjust(top=0.9)
        plt.savefig(self.out)
        
    def individual_scores_plot(self, logger:StructLogger, plot_title):
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
        plt.savefig(self.out)