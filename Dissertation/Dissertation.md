# Procedural content generation through the use of genetic algorithms (working title)
# Exploration vs exploitation - Investigating how different levels of novelty affect a population's overall fitness in NEAT algorithms
# How much is too much novelty?
## Abstract

##  Introduction
Procedural content generation (PCG) is a technique used in video game design and other creative fields, such as art and music. The technique allows creators to generate a near infinite amount of content based on a few rules, constraints, and parameters. This allows games to be highly replayable, feeling more unique and unpredictable. An example of effective PCG is in No Man’s Sky, a universe exploration game which can generate 2^64 (~18x10^18) worlds, each roughly 78.5 miles^2 (~203 sqkm). Comparing this to one the biggest non-generated games, ARMA 3, which has a map size of 270 sqkm, barely bigger than one of the quintillions of the worlds in No Mans’ Sky. 
https://www.nomanssky.com/
[https://www.gamespot.com/articles/making-sense-of-no-mans-skys-massive-universe/1100-6441344/]
https://arma3.com/
https://www.pcgamer.com/arma-3-map-might-be-bigger-than-you-ever-imagined/

PCG does come with some downsides, mainly sacrificing quality control. A traditional game would have game designers’ hand-designing all aspects of the game, meaning the output is predetermined and easily controllable. Since PCG is designed to be unending, it is impossible to completely quality check. Unlike other types of PCG, like art and music, the design of content in games is first and foremost based around playability, with artistic qualities being a secondary objective. Many existing machine learning PCG techniques allow for small variations in the content generated, which in a medium like music makes no difference to the overall quality, but small variations in game design can lead to the game being unplayable. (###Example about platformers?). But playability is not a binary scale, as [ref book] explains there is a "golden section" of game design which is the perfect balance novelty and familiarity, which is pleasing for our brains, therefore better to play.
https://books.google.co.uk/books?hl=en&lr=&id=3TAKAgAAQBAJ&oi=fnd&pg=PT3&dq=game+design&ots=EL229oE8MV&sig=t-DrjYbyrbdrTjhtZdF0mOGaNpM&redir_esc=y#v=onepage&q=game%20design&f=true

Another drawback to PCG is processing costs and generation speed. According to the steam hardware survey [ref], which is the most comprehensive hardware census and continuously updated, there is still a massive range in hardware capabilities being used, with strongest graphics card being ~2400% stronger than the lowest. The higher the hardware requirements for a game, the larger the proportion of users unable to run the game, alienating a portion of the customer base.
https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam
https://gpu.userbenchmark.com/Compare/Nvidia-RTX-3060-Ti-vs-Intel-UHD-Graphics-630-Desktop-Coffee-Lake-i5-i7/4090vsm356797

Outside of games, machine learning based PCG has made massive leaps forward. Models such as DALL-E 2 can generate completely unique photo-realistic images from a text description, and ChatGPT can generate text answers to almost any question. These models have been so successful due in large part to the enormous amount of data they have been trained on and the huge complexity of the models, which presents problems for PCG in games. For DALL-E/ChatGPT to generate whatever is requested, they needed to have seen something similar to base their response off of. This is problematic for game world generation, as there exists no dataset big or diverse enough to adequatly train these models. Large-scale deep learning models are also extremely computationally expensive, heavily affecting the hardware requirements.
https://openai.com/blog/chatgpt
https://openai.com/product/dall-e-2
https://arxiv.org/abs/1702.00539

Another issue is the idea of creativity. Models such as DALL-E and ChatGPT do produce unique outputs, but the output is based on many existing sources. For these models this is not a big problem because of the huge variety of training data which allows the models to draw on many sources to generate new content. Games on the other hand generally have the same recognisable styles and artifacts throughout the world. A model splicing these together could lead to a disjointed world and recognisable from other games, leading to lower immersion. 
https://www.ibm.com/watson/advantage-reports/future-of-artificial-intelligence/ai-creativity.html

The fundemental problem with PCG is the the *state-explosion* problem, where as the size of the generated content increases, the search space increases exponentially and brute force solutions become intractable. Since it is impossible to verify if a solution is the best solution, an approximate solution is required. One solution to this is through the use of genetic algorithms (GA). GA’s take inspiration from the biological process of natural selection and use it to evolve a solution to a problem by using a heuristic approach to move towards a better solution. Accomplished by removing ineffective algorithms from the population and allowing effective models to proceed and evolve further. Evolving a solution instead of training from past solutions comes with certain advantages. By having an algorithm which is controlled by the rules of a system rather than being trained on existing examples allows the model to come up with new unique solutions, rather than rehashing existing solutions. 
https://link.springer.com/content/pdf/10.1007/3-540-46002-0_19.pdf

In this paper I will be experimenting with NEAT, which is a type of genetic algorithm, and how the performance is affected by different levels of novelty. I will be investigating this within the context of the EvoCraft challenge, a PCG challenge for MineCraft. Theses concepts will be explained in more detail.

### Aims and Objectives
Aim: Investigate how varying levels of novelty in a population affects the population's overall fitness, within the context of the EvoCraft challenge.
Objectives:
    1. Investigate other competition entries for the EvoCraft challenge
    2. Research novelty techniques
    3. Train a basic model which can generate structures of arbitrary quality
    4. Implement basic novelty search
    5. Experiment with different levels of novelty
### Hypothesis
@@

## Background
### Minecraft
Minecraft is a 3D, open-world, sandbox, voxel-based video game. Each voxel, called a block, can be broken and replaced to build structures, allowing players to apply their creativity. Minecraft uses PCG and world seeds to create a unique world which is 3.6 billion blocks^2, allowing players virtually infinite space to explore and build. Because of the open-ended nature of the game and the simplicity of interactions with the world, Minecraft has become a platform for many AI challenges, including mineRL. This competition focused on an agent within Minecraft which has to complete a variety of tasks in an unknown environment. Because there is no one defined task, the algorithm has to be able to complete many smaller problems, with the eventual goal of improving research into general intelligence. 

https://www.sportskeeda.com/minecraft/how-big-minecraft-world
For interactions between Python and Minecraft world I will be using a Minecraft server with a Bukkit plugin called RaspberryJuice installed and the McPi Python library to communicate with it. Bukkit is a server modification tool with an API which allows users to easily create server plugins. The plugin converts Python commands into Java commands which can be processed by the Minecraft server. The McPi simply sends Python commands to the RaspberryJuice plugin.

## EvoCraft
To investigate genetic algorithms for PCG, I am working on the EvoCraft Challenge. The EvoCraft challenge brief is to create an open-ended algorithm which is capable of creating novel and increasingly complex structures in MineCraft. These algorithms have to be unending and should aim to diverge over time rather than slow down and become repetitive. One of the drawbacks of PGC was the lack of quality control, and the problem with infintely generating content becoming repetitive over time. This challenge aims to use evolving algorithms to keep generating content which keeps diverging and becoming more interesting.

### EvoCraft competition winners
#### Evocraft PCGNN *Michael Beukman, Matthew Kruger, Guy Axelrod, Manuel Fokam, Muhammad Nasir*
[ref] were came runners up in the EvoCraft competition with their endless city generator. Their approach broke a city down into the component 'levels', starting from the lowest level, the house and garden. To generate a house and garden they broke this down into 4 components: the house structure, roof, decorations, and garden. They then used a PCGNN (Procedural Content Generation using Neat and Novelty search) to generate each of these components. The house, roof, and decorations are all generated as 3d tilemaps to be placed in-world. The house consists of walls, empty space, and enterance, the roof consists of a design covering the area beneath it, and the decorations consists of decoration blocks filling floor space inside the house. The garden works slightly differently as it is a 2d tilemap covering an area with flowers, grass, and trees. They then used these component houses and gardens to create a town. A town is its own generated 2D tilemap of houses, gardens and roads @@image in appendix@@, where are road connects all the houses in the town. They then placed many towns together to create a city, which could grow endlessly. 
@@ Put about successes and drawbacks? About other details in paper
@@pic in appendix@@  
https://github.com/Michael-Beukman/Evocraft22_PCGNN

#### simple_minecraft_evolver *real_itu*
https://github.com/real-itu/simple_minecraft_evolver
#### Automated design of novel redstone circuits *Nicholas Guttenberg* 

https://github.com/GoodAI/EvocraftEntry

### Neural Networks
A neural network is a type of machine learning technique which is modelled after the human brain. A neural network consists of neurons, shown in figure @@. Each neuron has a weighted connection to other neurons, a bias, and an activation function  A neuron's value is calculated with the equation:
$$y=f(\sum_{i=0}^n w_{i}x_{i}+b)$$ 
Sigmoid activation function: $$f(x)=\frac{1}{(1+e^{-x})} $$
Where $n$ is the number of inputs, $b$ is the bias value, and $f(x)$ is the activation function. Shown is the the sigmoid activation function as an example, but many different activation functions exist. In a neural network many of these neurons are fully connected together to create a network of neurons, an example shown in figure @@. A typical neural network has an input layer, some hidden layers, and an output layer. To make a prediction from a neural network, some input values are given to the input layer neurons and those values are used to calculate the values in the next layer, then the next, until reaching the output layer. 

### NEAT + Novely Search
NeuroEvolution of Augmenting Topologies (NEAT) is type of genetic algorithm (GA) which mimics biological evolution to increase the complexity of neural networks. Traditional neural networks have a fixed structure of input, hidden, and output layers, usually fully connected by weights and biases. To improve the performance of the model the network is given examples of inputs and corresponding outputs, and techniques such as back propogation are used to update the network. NEAT works in a much different way. Instead of having a fixed structure inside the genome (the name for a network in NEAT context), the genome structure evolves over time. The genome is initialized with basic connections inside the genome, resulting in essentially random output. A population of individuals, each with a copy of the genome, is generated. This population then "evolves" through a process called "mutation". Each individual in the population will have random changes made to their internal structure. Each individual is then evaluated using a fitness function, which evaluates an output and assignes a score. The top individuals are then taken and the others are destroyed. The population is then recreated, to create a next "generation", using these top individuals and the mutation process is repeated. The goal of the process is, through random mutation, to move towards a network which can produce perfect output. NEAT, and other GA's, are especially useful when the size of the search space is extremely large and cannot be solved through exhaustive search techniques 
##Needs rewriting. Terminology is wrong and incomplete. Include other terms such as crossover use https://towardsdatascience.com/neat-an-awesome-approach-to-neuroevolution-3eca5cc7930f

There is however a problem with this evolution process, a population's fitness score can becomes stuck in a local maxima. This comes from a deeper ideological difference between biological evolution and genetic algorithms. The purpose of an GA is to reach this global maximum fitness, whereas biological evolution aims to both evolve individuals which will survive (high fitness), but will also spread out and diverse over generations. If a species does not diversify then the species is more vulnerable to diesases and changed to the environment. For an GA to get out of a local maxima it must first drop in fitness score before finding another gemone structure which would allow it to reach the global maxima. Traditional GA algorithms don't allows for this as any drop in fitness score would kill the individual and stop it from evolving further. Here we take a note from biological evolution and promote diversity, called novelty search. Novelty search gives a higher fitness to models, which may have a lower calculated fitness, but have genetically mutated and diverged from the other models and its parents. This allows models to drop in fitness and explore other methods of getting to a global maxima. 
https://www.biologicaldiversity.org/programs/biodiversity/elements_of_biodiversity/

### Reinforcement Learning PCG
Reinforcement learning (RL) is a type of machine learning composed of three key elements: an agent, the environment, a reward. RL uses a trial-and-error method with an agent interacting with an environment. An agent makes an action within and environment and will either be rewarded, if the the action was a positive action, and punished, if the action was negative. The eventual goal for a RL agent is to learn a policy, which is a mapping from input states to output actions. The agent starts with random actions and getting experience, in the form of state-action-reward, which is used to update the policy of the agent. Eventually the agent aims to maximise the cumulative reward signal by maintaining a balance between exploring, to learn new experiences, and exploitation, leveraging existing techniques.

RL gains many of the same benefits as genetic algorithms. Just like GA's, RL's require no large dataset and therefore don't have any of the intrinsic biases and creativity issues which come with it. There are some subtle differences between the two algorithms. RL can sometimes suffer trying to reach a global maxima in reward/fitness. Because RL is one agent learning over many iterations, if the agent chooses a strategy which works well, but only reaches a local maxima, it would have to unlearn that entire strategy and come up with a new one for it to reach a global maxima. A GA on the other hand has many individuals in a population, each which can explore their own routes (promoted through novelty search). Each route that ends in a local maxima will be killed in favour of a route which produces a global maxima, therefore having a higher likelihood of reaching the global maxima. While the RL model approach has the potential to be effective in this project, the properties of NEAT make it more desirable. 

https://pythonistaplanet.com/pros-and-cons-of-reinforcement-learning/

### General Adveserial Networks 
A General Adveserial Network (GAN) is type of deep learning network, composed of two neural networks: A generator and descriminator. The role of a generator is, once trained on the training data, to generate more examples which resemble the training data. The role of the descriminator is to tell the difference between examples from the training data and the examples generated by the generator. A common analogy for this the relationship between an art forger and art appraiser. The appraiser ensures that only the most convincing art forgeries survive and sell art, improving the quality of the forged art. 

GAN's have some powerful advantages which make them very powerful in certain situations. Since the training data is only used for examples, there is no need to label the data. Once a GAN is fully trained, both parts of the network can be used. A well trained generator can create very realistic and unique creative works, in many different styles. DALLE-2, a realistic image generator, uses a GAN-like network structure to create unique photo-realisic images. The descriminator can also be used to detecting abnormalities, for example medical imaging, quality control, and fraud detection. There are some isses with GAN's. They require vast amounts of training data and the wider the range of outputs being generated, the higher the size of the training data required. They are also a black box and very hard to reason why it came up with what it did, making it very hard to fix problems like diversity of output and garbled data. GAN's naturally lose detail from the input to the output, which makes the generated content lose some finer details. This is not a problem for creative works as some minor variation has no affect on the quality of the output, but does have an effect on PCG for games. Some minor changes to game generation can leave it unplayable and useless. The combination of not being able to generate at a very fine level, being a black box, and requiring huge amounts of training data, it can be very hard to reliable produce quality controlled content like games. Because of these disadvantages a GAN approach is unsuitable for this project.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8877944/
https://jonathan-hui.medium.com/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b

## Implementation
As previously mentioned, I will be experimenting with novelty in NEAT populations to create structures in MineCraft. This process can be broken down into three main objectives: genome creation, the fitness functions, and novelty experimentation. 
I have chosen to do this project in Python. While the competitors (Java, C++, R...) are much faster than the core Python modules, there are libraries, such as NumPy, which are written in C and are extremely quick. This means it can perform well on large datasets, making it a favourite for data scientists. Because it is a popular data science language, there is also a large collection of machine learning libraries supported.

### Genome creation
A genome is an individual within a population. It contains the information needed to create a neural network, which can be used to create structures. The aim is to create a genome which, given some configuration information (building dimensions, type of blocks...), can output a structure which follows the configuration instructions. To implement the NEAT algorithm, I am using a python library called python-neat. This library manages the encoding and mutation of the population, the only implementation that is required is giving the evoluation hyperparameters and evaluating the fitness of each individual.

#### Evolution hyperparameters
Hyperparameters are the parameters which control the learning process for each individual, such as mutation rate and network activation functions. One of these parameters is the input/output sizes of the genomes, which must be a fixed value for all individuals in a population at all times. The simplest way to use a genome to create a structure is to map each output neuron to a block to be placed in the structure. The disadvantage of this strategy is that since the output size is fixed, the size of the structure must be fixed, which limits the possible creativity and could easily become boring. A different strategy is to have the output not be a whole structure, but one block in the structure, and run the model many times. Using this technique the output size of the genome is each block possible to place, with the genome choosing the most likely block that should be placed. Unfortunately MineCraft contains over 1050 blocks/items, many of which cannot be placed or are dependant on other blocks around it, and there is no list of all structural blocks. Therefore I went through each blocks and made a list of each block which can be placed, which came out to 279 blocks. @@ put about stairs and wood blocks?@@. 

The inputs to the genome are now used to predict one block. @@Fig@@ shows a diagram for how this will be accomplished. Starting from the bottom layer and working upwards, the genome takes in the 17 surrounding blocks (apart from the blocks above as they are not created yet) and predicts the central block (in red). The building is also padded by one block, to give some values when predicting an edge block The model is also given the current xyz coordinates of the block being placed. This is required because the model has no memory, therefore it needs some value to know where the block is being placed relative to the rest of the structure. Also inputted to the model are the seed blocks which are used to give some control over which blocks are used in construction.@@ADD ANY MORE INPUTS@@

#### Fitness Functions
A fitness function is used to evalute the quality of a candidate solution, to help it reach the desired solution. According to @@ref@@ a fitness function should be:  
- Clearly defined: it should be easy to understand and provide meaningful insight into the performance.
- Intuitive: Better solutions should get a better score and visa-versa 
- Efficiently implemented: NEAT requires many iterations to evolve a good solution, so the fitness function should not be a bottleneck
- Sensitive: Should be able to distinguish between slightly better and slightly worse solutions to allow a gradual movement towards a better solution
https://towardsdatascience.com/how-to-define-a-fitness-function-in-a-genetic-algorithm-be572b9ea3b4

 To start writing functions, the components of the desired structure have to be broken down. An example building is shown in figure @@. When deciding fitness functions it is important to limit the scope of what can be expected, there is always more detail that can be added to the fitness. The purpose of a fitness function is to highlight the important parts of a structure and guide the genome towards the desired output, but leaving enough flexibility to allow for creativity. A balance between control and creativity 
The fitness functions I decided on were:
1. A bounding wall: There should be a complete wall around the building with no airgaps
2. Airspace: The interior space should be empty with only airblocks
3. A door: There should be a door on ground level on the structure to enter the structure
4. Seed blocks: The seed blocks should be consistently used over the structure
5. Symmetry: The building should be symmetrical on atleast one axis
6. @@ROOF FITNESS@


## Evaluation

## Future Work