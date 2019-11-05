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
import torch.nn.functional as functional
import torchvision.transforms as T

# own created classes
import CartPoleEnvManager

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#em = CartPoleEnvManager(device)
#em.reset()
#screen = em.render('rgb_array')

env = gym.make('CartPole-v0').unwrapped
env.reset()
screen = env.render('rgb_array')

plt.figure()
plt.imshow(screen)
plt.title('Non-processed screen example')
plt.show()