#%matplotlib inline
import gym
import math
import random
import numpy as numpy
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from intertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import torchvision.transforms as T