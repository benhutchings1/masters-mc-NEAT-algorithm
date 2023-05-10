from src.logger import MultiStructureLogger
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools
import math
from typing import Dict

class MultiStructurePlot():
    def __init__(self):
        self.conv_titles = {
            "high_novelty": "High Novelty",
            "all_novelty": "Full Novelty",
            "control": "Control",
            "low_novelty": "Low Novelty"
        }

    def plot_single_block_frequencies(self, ax, log, source):
        # Get Data
        block_freq = [[id, count] for id, count in log.block_counts.items()]
        block_freq = sorted(block_freq, key=lambda x: x[1], reverse=True)
        x = [str(b[0]) for b in block_freq]
        y = [b[1] for b in block_freq]
        # Plot on graph
        ax.bar(x, height=y)      
        # Decorate graph
        ax.set_xlabel("Block ID")
        ax.set_ylabel("Block frequency")
        ax.set_title(f"{self.conv_titles[source]} Block Distribution")
        
    def plot_block_frequencies(self, builders:Dict, out, filename):
        fig, axs = plt.subplots(2,2, figsize=(10,10))
        axs = [axs[i][j] for i in range(2) for j in range(2)]
        
        for ax, key in zip(axs, builders.keys()):
            self.plot_single_block_frequencies(ax, builders[key].logger, key)
        
        fig.set_tight_layout(0.5)
        plt.savefig(os.path.join(out, filename)) 
        
    
    def plot_single_scores(self, ax, logger, variable, key):
        data = [getattr(con, variable) for con in logger.constructions]
        ax.boxplot(data, vert=False)    
        ax.title.set_text(f"{self.conv_titles[key]} Score Distribution")
        ax.set_yticks([])
        
    
    def plot_all_scores(self, builders:Dict, out, filenames):
        for fname, data_type in zip(filenames, ["house_struct_score", "roof_struct_score"]):
            fig, axs = plt.subplots(2,2, figsize=(10,10))
            axs = [axs[i][j] for i in range(2) for j in range(2)]
            
            for ax, key in zip(axs, builders.keys()):
                self.plot_single_scores(ax, builders[key].logger, data_type, key)
            plt.savefig(os.path.join(out, fname)) 
   
    def average_of_many_lists(self, x):
        tot = 0
        size = 0
        for xi in x:
            tot += sum(xi)
            size += len(xi)
        return tot/size

    def plot_bar_var(self, loggers:Dict, out, filename):
        # Get variance data
        variances = {}
        for key, cb in loggers.items():
            var = cb.logger.get_building_variance()
            ind = []
            for model in var:
                ind.append(self.average_of_many_lists(model.values()))
            variances[key] = ind
        
        __, ax = plt.subplots(2, figsize=(10,10))
        roof_heights = []
        house_heights = []
        labels = []        
        
        for key, value in variances.items():
            house_heights.append(value[0])
            roof_heights.append(value[1])
            labels.append(key)
        
        ax[0].bar(range(len(labels)), house_heights, tick_label=labels)
        ax[0].set_title("House Models")
        ax[0].set_ylabel("Average Variance Score")
        ax[1].bar(range(len(labels)), roof_heights, tick_label=labels)
        ax[1].set_title("Roof Models")
        ax[1].set_ylabel("Average Variance Score")
        
        plt.suptitle("Comparisons of average building variance")
        plt.savefig(os.path.join(out, filename)) 
    
    def plot_gen_time_bar(self, loggers:Dict, out, filename):       
        times = {key:val.logger.get_avg_generation_time() for key, val in loggers.items()}
        fig, ax = plt.subplots(2, figsize=(10,10))
        roof_heights = []
        house_heights = []
        labels = []        
        
        for key, value in times.items():
            house_heights.append(value[0])
            roof_heights.append(value[1])
            labels.append(key)
        
        ax[0].bar(range(len(labels)), house_heights, tick_label=labels)
        ax[0].set_title("House Models")
        ax[0].set_ylabel("Average Generation Time per structure (s)")
        ax[1].bar(range(len(labels)), roof_heights, tick_label=labels)
        ax[1].set_title("Roof Models")
        ax[1].set_ylabel("Average Generation Time per structure (s)")
        
        plt.suptitle("Comparisons of Average Generation time/structure")
        plt.savefig(os.path.join(out, filename)) 
        