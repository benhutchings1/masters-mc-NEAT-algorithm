from src.visualise import struct_plot
from src.logger import StructLogger
import os
os.mkdir("graphs/")
for source in ["roof", "house"]:
    control_loggers = [
        StructLogger(f"data/control/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"data/control/iter2/{source}/struct", "struct_log.csv", None, False)  
    ]
    dynamic_loggers = [
        StructLogger(f"data/dynamic/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"data/dynamic/iter2/{source}/struct", "struct_log.csv", None, False)
    ]
    high_novelty = [
        StructLogger(f"data/high_novelty/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"data/high_novelty/iter2/{source}/struct", "struct_log.csv", None, False)
    ]
    low_novelty = [
        StructLogger(f"data/low_novelty/iter1/{source}/struct", "struct_log.csv", None, False),
        StructLogger(f"data/low_novelty/iter2/{source}/struct", "struct_log.csv", None, False)
    ]   

    struct_plot.StructPlot(out=f"graphs/control/{source}/").average_plot(
        ["Iter 1", "Iter 2"],
        control_loggers,
        "Control"   
    )

    struct_plot.StructPlot(out=f"graphs/dynamic/{source}/").average_plot(
        ["Iter 1", "Iter 2"],
        dynamic_loggers,
        "Dynamic"   
    )

    struct_plot.StructPlot(out=f"graphs/high_novelty/{source}/").average_plot(
        ["Iter 1", "Iter 2"],
        high_novelty,
        "High Novelty"   
    )

    struct_plot.StructPlot(out=f"graphs/low_novelty/{source}/").average_plot(
        ["Iter 1", "Iter 2"],
        low_novelty,
        "Low Novelty"   
    )                           