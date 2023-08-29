from src.visualise import struct_plot
from src.logger import StructLogger
import os

out = "data"

try:
    os.mkdir("graphs/")
except:
    pass

for source in ["roof", "house"]:
    # Get loggers for all experiments
    control_loggers = [
        StructLogger(f"{out}/control/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"{out}/control/iter1/{source}/struct", "struct_log.csv", None, False)  
    ]
    high_novelty = [
        StructLogger(f"{out}/high_novelty/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"{out}/high_novelty/iter2/{source}/struct", "struct_log.csv", None, False)
    ]
    low_novelty = [
        StructLogger(f"{out}/low_novelty/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"{out}/low_novelty/iter2/{source}/struct", "struct_log.csv", None, False)
    ]   
    
    # Plot control graphs
    struct_plot.StructPlot(out=f"graphs/control/").plot(
        ["Run 1", "Run 2"],
        control_loggers,
        "Control",
        f"control_{source}",
        const=0.3
    )
    print("Control graphs completed")

    # Plot high novelty graphs
    struct_plot.StructPlot(out=f"graphs/high_novelty/").plot(
        ["Run 1", "Run 2"],
        high_novelty,
        "High Novelty",
        f"high_nov_{source}"
    )
    print("High novelty graphs completed")
     
    # Plot low novelty graphs
    struct_plot.StructPlot(out=f"graphs/low_novelty/").plot(
        ["Run 1", "Run 2"],
        low_novelty,
        "Low Novelty",
        f"low_nov_{source}"   
    )    
    print("Low novelty graphs completed")                       
