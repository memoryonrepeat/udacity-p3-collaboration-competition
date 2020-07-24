### Overview 

This repo contains my submission for Udacity Deep Reinforcement Learning course - Project 2.

The project is about training 2 agents to try to keep the ball in play for as long as possible in a tennis game.

A reward of +0.1 is provided whenever an agent hits the ball over the net.

A reward of -0.1 is provided whenever an agent lets the ball hit the ground or hits the ball out of bound.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 

Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic and is considered successful when the agents get an average score of 0.5 over 100 consecutive episodes, after taking the maximum over both agents. To be precise:


- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.

- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Setup

The agent is implemented in `ddpg_agent.py`.

There is a `Tennis.ipynb` file with all dependency requirements placed on the top cell, including the agent file.

These dependencies are readily available on Udacity workspace. On local, any missing dependency can be installed via `pip install <dependency name>`.

Models are defined in-line within the notebook.

The environment can be obtained from the [official Udacity repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started).

Once downloaded, unzip and place it in the same folder with the notebook, and change the file path accordingly to point to the environment file.

Now setup is finished. Running all the cells within `Tennis.ipynb` will walk you through the training.

Successful model weights are saved in the `checkpoint_actorX.pth` and `checkpoint_criticX.pth` files, with X = [0,1] corresponding to the first or second agent.

Note that to be able to see the agents while training, the notebook needs to be run on local.


### Training

This can be found in the `REPORT.md` file
