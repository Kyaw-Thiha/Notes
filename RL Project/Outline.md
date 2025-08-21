## Outline
Creating a practical Reinforcement-Learning based bot for World War-2 themed hex-based, turn-based game

## Background Research
No turn-based hex strategy games have truly implemented reinforcement learning as their AI bot.

There was only 1 research paper which attempted to research into this  
https://arxiv.org/html/2502.13918v1

[[Background Research| Read More]]

## Research Question
To what extent can hierarchical RL with curriculum learning enable agents to transfer policies learned on small hex-based maps to larger and more complex game environments?

## Unique Selling Point
Here are the reasons why we can have a higher chance of achieving our goal compared to previous work
- Optimizing game design for RL training (through reducing action space, making it 1unit per turn like chess)
- Training on layered strategic goals
  Eg:
  Tactical Level - Concentrate fire on enemy unit
  Operation Level - Breakthrough, Flank, Encircling
  Strategic Level - Capture a city

## Game Design
The game will be designed to optimise the learning capabilities of reinforcement learning agent.
[[Game Design| Read More]]

## Model Architectures
The main training strategy we are planning to try out is to train on layered strategic goals.
The main idea is to teach the agent in a very much similar manner to the way an actual military organization would plan to achieve its goal.

[[Model Architecture |Read More]]

## Fail-Safe Nature
One important note about this research project is that given that the closest research paper has only managed to achieve optimal results for 5x5, and but struggled at 10x10 mean that there is a no research paper to 'fall back onto'
Moreover, we will be using a completely different approach to it.

Compared to it, we will be trying more 'realistic' game effects like more unit variety, combat effects, and objectives, and terrain variety.
So, getting any results based on these would still be worth to put into a research paper.

Moreover, our hierarchical nature mean that even if we manage to train our model to optimize on 'tactical' level, we could make the other models to be 'heuristic' (rule-based) model and still make it into a presentable product.

[[Analysis on Closest Paper]]

## Learning Resources
- RL Theory Textbook - [Link](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- RL Tutorial - [Link](https://spinningup.openai.com/en/latest/)
- AlphaZero Research Paper - [Link](https://arxiv.org/pdf/1712.01815)
