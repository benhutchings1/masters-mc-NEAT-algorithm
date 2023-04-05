# Procedural content generation through the use of genetic algorithms (working title)
## Abstract

##  Introduction
Procedural content generation (PCG) is a technique used in video game design and other creative fields, such as art and music. The technique allows creators to generate a near infinite amount of content based on a few rules, constraints, and parameters. This allows games to be highly replayable, feeling more unique and unpredictable. An example of effective PCG is in No Man’s Sky, a universe exploration game which can generate 2^64 (~18x10^18) worlds, each roughly 78.5 miles^2 (~203 sqkm). Comparing this to one the biggest non-generated games, ARMA 3, which has a map size of 270 sqkm, barely bigger than one of the quintillions of the worlds in No Mans’ Sky. 

PCG does come with some downsides, mainly sacrificing quality control. A traditional game would have game designers’ hand-designing all aspects of the game, meaning the output is predetermined and easily controllable. Since PCG is designed to be unending, it is impossible to quality check everything. This can lead to some areas of a game being unplayable or the content eventually becoming repetitive. Another drawback is the processing costs. Since the content is being generated, instead of being read from the game files, there is a much larger overhead. 

Outside of games, machine learning based PCG has made massive leaps forward. Models such as DALL-E 2  can generate completely unique photo-realistic images from a text description, and ChatGPT can generate text answers to almost any question. These models have been so successful due to the enormous amount of data they have been trained on and the huge complexity of the models, which present problems for PCG in games. For DALL-E/ChatGPT to generate whatever is requested, they need to have seen something similar to base their response off of. For game world generation, there exists no dataset big or diverse enough to train these models. Deep learning models are also extremely computationally expensive, so making very large worlds can be too intense for many devices. 
Another issue is the idea of creativity. Models such as DALL-E and ChatGPT do produce unique outputs, but the output is based on many existing sources. For these models this is not a big problem because of the huge variety of training data which allows the models to draw on many sources to generate new content. Games on the other hand generally have the same recognisable styles and artifacts throughout the world. A model splicing these together could lead to a disjointed world and recognisable from other games, leading to lower immersion. 

One solution to this is through the use of genetic algorithms (GA). GA’s take inspiration from the biological process of natural selection and use it to evolve a solution to a problem. Evolving a solution instead of training from past solutions comes with certain advantages. By having an algorithm which is controlled by the rules of a system rather than being trained on existing examples allows the model to come up with new unique solutions, rather than rehashing existing solutions. This evolution method also means the model doesn't require any previous data, sidestepping the problems with having no large dataset.

https://www.nomanssky.com/
[https://www.gamespot.com/articles/making-sense-of-no-mans-skys-massive-universe/1100-6441344/]
https://arma3.com/
https://www.pcgamer.com/arma-3-map-might-be-bigger-than-you-ever-imagined/
https://openai.com/product/dall-e-2

### Aims and Objectives
Aim: 


## Background
### Minecraft
Minecraft is a 3D, open-world, sandbox, voxel-based video game. Each voxel, called a block, can be broken and replaced to build structures, allowing players to apply their creativity. Minecraft uses PCG and world seeds to create a unique world which is 3.6 billion blocks^2, allowing players virtually infinite space to explore and build. Because of the open-ended nature of the game and the simplicity of interactions with the world, Minecraft has become a platform for many AI challenges, including mineRL. This competition focused on an agent within Minecraft which has to complete a variety of tasks in an unknown environment. Because there is no one defined task, the algorithm has to be able to complete many smaller problems, with the eventual goal of improving research into general intelligence. 

https://www.sportskeeda.com/minecraft/how-big-minecraft-world
### EvoCraft Challenge
The EvoCraft challenge brief is to create an open-ended algorithm which is capable of creating novel and increasingly complex structures. These algorithms have to be unending and should aim to diverge over time rather than slow down and become repetitive. One of the drawbacks of PGC was the lack of quality control, and the problem with infintely generating content becoming repetitive over time. This challenge aims to use evolving algorithms to keep generating content which is interesting.

### NEAT + Novely Search
NeuroEvolution of Augmenting Topologies (NEAT) is a genetic algorithm (GA) which uses biological evolution to increase the complexity of neural networks. Traditional neural networks have a fixed structure of input, hidden, and output layers, usually fully connected by weights and biases. To improve the performance of the model the network is given examples of inputs and corresponding outputs, and techniques such as back propogation are used to update the network. NEAT works in a much different way. Instead of having a fixed structure inside the genome (the name for a network in NEAT context), the genome structure evolves over time. The genome is initialized with basic connections inside the genome, resulting in essentially random output. A population of individuals, each with a copy of the genome, is generated. This population then "evolves" through a process called "mutation". Each individual in the population will have a/many random changes made to their internal structure. Each individual is then evaluated using a fitness function, which evaluates an output and assignes a score. The top individuals are then taken and the others are destroyed. The population is then recreated, to create a next "generation", using these top individuals and the mutation process is repeated. The goal of the process is, through random mutation, to move towards a network which can produce perfect output. 

There is however a problem with this evolution process, a population's fitness score can becomes stuck in a local maxima. Moving out of this spot would require generations with lower fitness scores, but after some generations the population fitness would increase to reach a global maxima. This is a problem because any individual which drops in fitness would be eliminated, stopping the population from diversing. A solution to this is incorperating novelty search. In novelty serach not only does the fitness function evaluate the output of a model it also measures how diverse the genetic structure of an individual is compared to the population, and uses this to boost its fitness score. This promotes individuals which branch out and evolve different structures, hopefully improving the chances of making a network which reaches a perfect fitness score.

## Method

## Evaluation

## Future Work