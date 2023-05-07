from src.visualise import struct_plot
from src.logger import StructLogger
import os

try:
    os.mkdir("graphs/")
except:
    pass

for source in ["roof", "house"]:
    control_loggers = [
        StructLogger(f"data/control/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"data/control/iter1/{source}/struct", "struct_log.csv", None, False)  
    ]
    high_novelty = [
        StructLogger(f"data/high_novelty/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"data/high_novelty/iter2/{source}/struct", "struct_log.csv", None, False)
    ]
    low_novelty = [
        StructLogger(f"data/low_novelty/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"data/low_novelty/iter2/{source}/struct", "struct_log.csv", None, False)
    ]   

    struct_plot.StructPlot(out=f"graphs/control/").plot(
        ["Run 1", "Run 2"],
        control_loggers,
        "Control",
        f"control_{source}",
        const=0.3
    )
    print("Control graphs completed")

    struct_plot.StructPlot(out=f"graphs/high_novelty/").plot(
        ["Run 1", "Run 2"],
        high_novelty,
        "High Novelty",
        f"high_nov_{source}"
    )
    print("High novelty graphs completed")

    struct_plot.StructPlot(out=f"graphs/low_novelty/").plot(
        ["Run 1", "Run 2"],
        low_novelty,
        "Low Novelty",
        f"low_nov_{source}"   
    )    
    print("Low novelty graphs completed")                       
