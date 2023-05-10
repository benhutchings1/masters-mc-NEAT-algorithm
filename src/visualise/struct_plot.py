from src.logger import StructLogger
import matplotlib.pyplot as plt 
import os
import numpy as np
import math
from typing import List
from scipy.ndimage import uniform_filter1d

class StructPlot(StructLogger):
    def __init__(self, out="out/struct"):       
        self.primary_color = "orange"
        self.secondary_color = "grey"
        # Make output files
        if out is not None:
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
        # ax.plot(
        #     range((len(data))),
        #     max_data,
        #     label="Max Structure Score",
        #     alpha=0.75
        # )
        # ax.plot(
        #     range((len(data))),
        #     min_data,
        #     label="Min Structure Score",
        #     alpha=0.75
        # )
        ax.fill_between(range((len(data))),
                        max_data,
                        min_data,
                        alpha=0.8,
                        color=self.secondary_color,
                        label="Max-Min gap")
        ax.plot(
            range((len(data))),
            avg_data,
            label="Avg Struct Score",
            color=self.primary_color
        )
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Structure Score")
        ax.legend()
    
    def plot(self, titles, loggers:List[StructLogger], super_plot_title, filename, const=0, max_read=None):
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
            self.__plot_values(ax[0], loggers[0].get_scores(max_read=max_read), titles[0])
        else:
            # Fill graph
            for i in range(len(loggers)):
                self.__plot_values(ax[i], loggers[i].get_scores(max_read=max_read[i]), titles[i], const=const[i])
                
            
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
        
    def highest_moving_average(self, logger:StructLogger, window_size, const=0):
        data = [np.average(x.astype(float)) for x in logger.read_file()]
        return np.max(uniform_filter1d(data, window_size))-const

    def bar_plot_both(self, house, roof, houselabels, rooflabels, filename):
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        fig.tight_layout()
        plt.subplots_adjust(left=0.08, top=0.85, bottom=0.1)
        for a, d, lbl, title, lim in zip(ax, [house, roof], [houselabels, rooflabels], ["House Model", "Roof Model"], [0.35, 0.45]):
            self.bar_plot(
                a, d, lbl, title, lim
            )
        ax[0].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False)         # ticks along the top edge are off
        ax[1].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False)         # ticks along the top edge are off
        
        plt.show()
        plt.suptitle("Max Average Structure Score Across Populations (20 generation window)")
        fig.supxlabel("Novelty types (Both runs)")
        plt.savefig(os.path.join(self.out, filename))
        
    def bar_plot(self, ax, data, xlabels, title, lim):
        bar_width = 0.25
        xticks = list(range(0, len(data), 1))
        xticks = [[a-(2*bar_width/3), a+(2*bar_width/3)] for a in xticks]
        xticks = [item for sublist in xticks for item in sublist] 
        data = [item for sublist in data for item in sublist]
        xlabels = [[lbl, ""] for lbl in xlabels]
        xlabels = [item for sublist in xlabels for item in sublist]
        colors = ["orange", "blue"] * len(data)
        ax.bar(xticks, data, width=bar_width, align="center", tick_label=xlabels, color=colors)
        ax.set_ylabel("Max Average Structure Score")
        ax.set_title(title)
        ax.set_ylim(bottom=lim)
        
        
        
        
        