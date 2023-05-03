from src.visualise import struct_plot
from src.logger import StructLogger

loggers = [
    StructLogger("logs/roof/struct/", "struct_log.csv", None, False)
]


struct_log = struct_plot.StructPlot().plot(
    ["Dynamic Novelty"],
    loggers
)

