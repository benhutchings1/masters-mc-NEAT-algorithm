# Generation Of Diverse Minecraft Structures Using Evolutionary Algorithms

## Introduction

My MSc dissertation project looks at using NeuroEvolution of Augmenting Topologies (NEAT)[1] to attempt the MineCraft EvoCraft challenge[2] and the effect of novelty on the performance. The EvoCraft challenge brief is to create an open-ended algorithm which is capable of creating novel and increasingly complex structures in MineCraft. For my attempt, houses are infitely dynamically created which is accomplished with 2 models, one to create the structure of the house and the other to create the roof. To test this novelty, different structure/roof models are trained using different levels of novelty, then the characteristics of the models are compared to determine the effects. A further description of the implementation, training, and results analysis can be found in ```Dissertation/dissertation.pdf```.

## Usage

The project is composed of two parts, model training and content generation, which are run independently. Each part comes with data logging and analysis. Model training includes training analysis to show improvement over time and content generation analysis shows the quality of generation (single structure and multistructure).

### Starting Python Environment

A virtual environment is required, the dependancies of which can be found in ```requirements.txt```.

### Model Training And Visualiation

#### Model Training

There are two models (structure/house and roof) which are trained separately. Training is composed of 3 experiments (control/no novelty, low novelty, high novelty) and for each experiment two iterations are done.
2 models x 3 experiments x 2 iterations = 12 experiment runs

To implement the learning, the neat-python library is used. The configuration of this learning is controlled by the config files in ```config/House-NEAT-config``` and ```config/Roof-NEAT-config```. Other configuration options can be found in ```CONFIG.py```. To run the full experiment, run ```run_full_training.py```. The output location of the checkpoints and the generated data can be controlled by ```DATA_OUTPUT_PATH``` in ```CONFIG.py```. If the execution of the training is interrupted, it can be restarted using a checkpoint for each model. This is done by calling ```run_from_checkpoint(checkpoint path)``` instead of ```run()```, using the same configuration as the checkpoint. The output location for the checkpoints are```DATA_OUTPUT_PATH/[experiment type]/[iteration]/[model type]/checkpoints```

Training output data (for each experiment run):

- Checkpoints (used for generation and recovery)
- Novelty (CSV file with changes in novelty over the generations)
- Stats (how long each generation took)
- Structure Scores (The scores of each individual in each generation according to the fitness functions relevant to the model)

#### Visualisation

Once training is completed the training performance can be visualised using ```visualise.py```. The location of the graph data is controlled by ```VISUALISATION_OUTPUT``` in ```CONFIG.py```. This analysis shows how the average structure score improves over time for all experiment runs as a line graph

### Placement and Analysis

Placement involves taking a checkpoint from each experiment run and use it to generate structures. The placement of these structures within a MC server is optional. The structures generated are then analysed. The analysis data is placed in ```DATA_OUTPUT_PATH``` and the analysis graphs are  

#### Placement

The placement and analysis are done together. The checkpoints used to generate content are automatically gathered by taking the highest generation checkpoint. To specify the generation change the IF statement and update the run_config variable.

### References

1. K. O. Stanley and R. Miikkulainen, “Evolving Neural Networks through Augmenting Topologies,” Evol Comput, vol. 10, no. 2, pp. 99–127, 2002, doi: 10.1162/106365602320169811.
2. D. Grbic, R. B. Palm, E. Najarro, C. Glanois, and S. Risi, “EvoCraft: A New Challenge for Open-Endedness.” 2020.