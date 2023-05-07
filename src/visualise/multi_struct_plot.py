from src.logger import MultiStructureLogger
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools

class MultiStructurePlot():
    def __init__(self, multistructure_log:MultiStructureLogger):
        self.log = multistructure_log

    def plot_block_frequencies(self, out):
        # Get Data
        block_freq = [[id, count] for id, count in self.log.block_counts.items()]
        block_freq = sorted(block_freq, key=lambda x: x[1], reverse=True)
        x = [str(b[0]) for b in block_freq]
        y = [b[1] for b in block_freq]
        # Plot on graph
        fix, ax = plt.subplots(1)
        ax.bar(x, height=y)
        
        # Try make output directory
        try:
            os.mkdir(out)
        except:
            pass
        # Decorate graph
        ax.set_xlabel("Block ID")
        ax.set_ylabel("Block frequency")
        
        plt.savefig(out)
    
    def plot_scores(self, out):
        house_scores = [con.house_struct_score for con in self.log.constructions]
        roof_scores = [con.roof_struct_score for con in self.log.constructions]
        
        fig, ax = plt.subplots(2)
        ax[0].boxplot(house_scores, vert=False)
        ax[1].boxplot(roof_scores, vert=False)
        
        # Try make output directory
        try:
            os.mkdir(out)
        except:
            pass
        # Decorate graph
        ax[0].title.set_text("House Structure Score Distribution")
        ax[1].title.set_text("Roof Structure Score Distribution")
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        fig.set_tight_layout(0.5)
        plt.savefig(out)       
            