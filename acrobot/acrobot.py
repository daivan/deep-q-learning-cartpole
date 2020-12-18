#%matplotlib inline
import gym

import math
import random
import numpy as numpy
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from AcroBotEnvManager import AcroBotEnvManager
from EpsilonGreedyStrategy import EpsilonGreedyStrategy
from Agent import Agent
from ReplayMemory import ReplayMemory
#from DQN import DQN
#from Experience import Experience
#from QValues import QValues


# initiate variables

batch_size = 256
gamma = 0.999

eps_start = 1
eps_end = 0.01
eps_decay = 0.001

target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

# Load a pytorch device with either cuda (GPU) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate the strategy for exploit or explore and how its moving twards exploit
strategy = EpsilonGreedyStrategy(eps_start,eps_end,eps_decay)

# Handles the environment, the game screen it self and what moves you can do and if you are done.
em = AcroBotEnvManager(device)

# Create the agent that is suppose to play the game
agent = Agent(strategy, em.num_actions_available(),device)

# Handles DQN memory of all the games youve played. Database of your previous games
memory = ReplayMemory(memory_size)

env = gym.make('Acrobot-v1')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()