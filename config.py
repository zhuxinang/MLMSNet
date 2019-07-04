# -*- coding:utf-8 -*-
import os
import torch
from torchvision import models

from numpy import random
import time, pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np


# from gan import *from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
from torch.autograd import Variable
#from visdom import Visdom


RE_LEARNING_RATE = 3e-5

import time
import torch
U_LEARNING_RATE=1e-4
NN =8
re_SIZE=[(64,64),(32,32),(16,16),(8,8)]
re_SIZE_2 = [(32,32),(128,128)]
BATCH_SIZE =18
NUM_WORKERS = 8
NUM_EPOCHS = 200
SIZE2 =(256,256)
SIZE3 =(350,350)
D_LEARNING_RATE =2e-4



IMG_SIZE = (256, 256)
LABEL_SIZE = (256, 256)


