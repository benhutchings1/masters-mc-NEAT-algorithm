# Title
##  Introduction (~2-3 pages)
- Motivation and rationale
- Procedural content generation, open ended algorithms
    - Games
        - Good design can make or break game
        - Infinite content generation
        - Increased replayability
        - Reduced file sizes for games
        - Lower budgets for games
        - No mans sky
    - Art generation/Urban modelling (non games uses)
- Neural nets can only build off existing information
- Generates structures with no knowledge of existing structures
- Dont have to curate massive database
- 
- Aims and objs
    - Why
        - What is the problem that needs solving
    - How
        - What is the approach
        - What methods will be used
    - What
        - What steps need to be taken
    - Milestones
        - Experimenting with system behaviours

### Aim and Objectives
Aim: 
Objectives  
1. Experiment with other creations



## Related Work (~6-8 pages)
- PGC 
    - Previous techniques
- Evocraft
    - https://evocraft.life/
    - gRPC calls framework 
- Minecraft
    - Project Malmo & MineRL
    - Minecraft AI Settlement Generation Challenge
    - Voxel space
- NEAT
    - Fitness functions
    - Novelty Search
        - Critical factors in novelty search https://dl.acm.org/doi/abs/10.1145/2001576.2001708?casa_token=m49nq4A5exsAAAAA:iQROZsm1KBg_uBPa-lQPmB36D32dhf7yPWxT4jIrIPMSG1SiQ2tYQHDWL2MpIUhdFLxAbM6r-0vDSBw

    - https://dl.acm.org/doi/pdf/10.1145/2001576.2001606
-  Evocraft22_PCGNN
    - https://github.com/Michael-Beukman/Evocraft22_PCGNN/tree/main/src/analysis/proper_experiments/v400
    - i am focusing on a much smaller scale generation, they focused on large scale diversity?
- GAN
    - Can be much bigger models

## Method (~10-12 pages)
- End goal
    - Requirements
- Evaluation of language choice
- NEAT python
    - Algorithmic fitnesss functions
    - Novelty search
        - Put through activation function
        - dist fuction
        - Archived and NN networks 
            - Might cycle through evolutinoary behaviours
        - Experiment
            - Test with no novelty
            - Test with low novelty
            - test medium novelty
            - test high novelty
            - Dynamic diversity? 
                - Inverse differential of fitness
- Block placement   


## Evaluation (~5-7 pages)
- Minecraft performance 
- Fitness over time
    - Max, mean, min, fitness
    - Performance scaling with size of building generated
- Novelty can be high but has to be human novelty which is high
- 
## Future Work
- Having some memory based NN?
- Feature network (adding chimneys, balconies....)
## References (~1-2 pages)
https://subscription.packtpub.com/book/game-development/9781785886713/1/ch01lvl1sec14/the-drawbacks-of-procedural-generation




TODO
Update with new aim
Update NEAT description to show description of cross over mutate....