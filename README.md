# Generation Of Diverse Minecraft Structures Using Evolutionary Algorithms
## Introduction
My MSc dissertation project looks at using NeuroEvolution of Augmenting Topologies (NEAT)[1] to attempt the MineCraft EvoCraft challenge[2] and the effect of novelty on the performance. The EvoCraft challenge brief is to create an open-ended algorithm which is capable of creating novel and increasingly complex structures in MineCraft. For my attempt, houses are infitely dynamically created which is accomplished with 2 models, one to create the structure of the house and the other to create the roof. To test this novelty, different house/roof models are trained using different levels of novelty, then the characteristics of the models are compared to determine the effects. A further description of the implementation, training, and results analysis can be found in ```Dissertation/dissertation.pdf```. 

## Usage
The project is composed of two parts, model training and house placement/analysis, which are run independently.
### Starting Python Environment
To develop the implementation a Pipenv virtual environment was used, the dependancies of which can be found in ```requirements.txt```.

### Model Training
To implement the learning, the neat-python library is used. The configuration of this learning is controlled by the config files in ```config/House-NEAT-config``` and ```config/Roof-NEAT-config```. To run the full experiment, run ```run_full_training.py```. The default output location of the checkpoints and the generated data is ```~/data/```, but this can be changed using the ```out``` variable. If the execution of the training is interrupted, it can be restarted using a checkpoint for each model. This is done by calling ```run_from_checkpoint(checkpoint path)``` instead of ```run()```, using the same configuration as the checkpoint. Once training is completed the training performance can be visualised using ```visualise.py```. The location of the graph data should be entered using the ```out``` variable.

### Placement and Analysis


1. K. O. Stanley and R. Miikkulainen, “Evolving Neural Networks through Augmenting Topologies,” Evol Comput, vol. 10, no. 2, pp. 99–127, 2002, doi: 10.1162/106365602320169811.
2. D. Grbic, R. B. Palm, E. Najarro, C. Glanois, and S. Risi, “EvoCraft: A New Challenge for Open-Endedness.” 2020.