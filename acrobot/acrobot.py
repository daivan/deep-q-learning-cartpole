#%matplotlib inline
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
from DQN import DQN
from Experience import Experience
from QValues import QValues


# initiate variables

batch_size = 50
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


#  DQN policies
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params = policy_net.parameters(), lr=lr)



def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episiode", len(values), "\n", moving_avg_period, "episode moving avg:", values[-1])

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    
    else:
        moving_avg= torch.zeros(len(values))
        return moving_avg.numpy()


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

episode_durations = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()
    
    score = 0
    
    for timestep in count():
        #print(timestep)
        action = agent.select_action(state, policy_net)
        #reward = em.take_action(action)
        reward, real_reward = em.take_action(action)
        score = score + real_reward
        next_state = em.get_state()
        
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):

            experiences = memory.sample(batch_size)

            states, actions, rewards, next_states = extract_tensors(experiences)

   
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + reward

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if em.done:
        #if timestep>100:
            #print('do we ever get here?')
            episode_durations.append(score)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

em.close()
