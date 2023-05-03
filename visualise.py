from src.visualise import struct_plot
from src.logger import StructLogger

control_loggers = [
    StructLogger("data/control/iter1/house/struct", "struct_log.csv", None, False),
    StructLogger("data/control/iter2/house/struct", "struct_log.csv", None, False)  
]
dynamic_loggers = [
    StructLogger("data/dynamic/iter1/house/struct", "struct_log.csv", None, False),
    StructLogger("data/dynamic/iter2/house/struct", "struct_log.csv", None, False)
]
high_novelty = [
    StructLogger("data/high_novelty/iter1/house/struct", "struct_log.csv", None, False),
    StructLogger("data/high_novelty/iter2/house/struct", "struct_log.csv", None, False)
]
low_novelty = [
    StructLogger("data/low_novelty/iter1/house/struct", "struct_log.csv", None, False),
    StructLogger("data/low_novelty/iter2/house/struct", "struct_log.csv", None, False)
]   

# struct_plot.StructPlot(out="out/control/").plot(
#     ["Iter 1", "Iter 2"],
#     control_loggers,
#     "Control"   
# )

# struct_plot.StructPlot(out="out/dynamic/").plot(
#     ["Iter 1", "Iter 2"],
#     dynamic_loggers,
#     "Dynamic"   
# )

# struct_plot.StructPlot(out="out/high_novelty/").plot(
#     ["Iter 1", "Iter 2"],
#     high_novelty,
#     "High Novelty"   
# )

# struct_plot.StructPlot(out="out/low_novelty/").plot(
#     ["Iter 1", "Iter 2"],
#     low_novelty,
#     "Low Novelty"   
# )

sp = struct_plot.StructPlot()
sp.individual_scores_plot(low_novelty[1], "dynamic")