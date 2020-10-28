import io
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math


# Set vmin to be the smallest plausible value for the optimal minimizer, and vmax to be the maximum plausible value. If these values are not correct then regret is not guaranteed. 
class SwapMinimizer:
    def __init__(self, vmin=0.0, vmax=1.0, num_step=100):
        self.ab_list = np.zeros(num_step, dtype=np.float)
        self.b2_list = np.zeros(num_step, dtype=np.float)
        self.cur_action = 0
        self.vmin = vmin
        self.vmax = vmax
        self.num_step = num_step
        
    def get_pred(self):
        action_list = []
        while True:
            if self.b2_list[self.cur_action] == 0:
                return self.to_continuous(self.cur_action)

            new_pred = self.ab_list[self.cur_action] / self.b2_list[self.cur_action]
            if self.to_discrete(new_pred) == self.cur_action:
                return new_pred
            self.cur_action = self.to_discrete(new_pred)

            if self.cur_action in action_list:
                self.cur_action = np.random.choice(action_list[action_list.index(self.cur_action):])
                return self.ab_list[self.cur_action] / self.b2_list[self.cur_action]
            action_list.append(self.cur_action)
        
    def to_discrete(self, val):
        return max(min(int(math.floor((val - self.vmin) / (self.vmax - self.vmin) * self.num_step)), self.num_step - 1), 0)
    
    def to_continuous(self, index):
        return float(index) / self.num_step * (self.vmax - self.vmin) + self.vmin
    
    def set_outcome(self, a, b):
        self.ab_list[self.cur_action] += -a * b
        self.b2_list[self.cur_action] += b * b

        
class AvgMinimizer:
    def __init__(self, vmin=0.0, vmax=1.0, num_step=100):
        self.asum = 0.0
        self.bsum = 0.0
    
    def get_pred(self):
        if self.bsum == 0:
            new_pred = 0.0
        else:
            new_pred = self.asum / self.bsum
        return new_pred
    
    def set_outcome(self, a, b):
        self.asum += -a
        self.bsum += b
        
        
class NaiveMinimizer:
    def __init__(self, vmin=0.0, vmax=1.0, num_step=100):
        self.ab = 0.0
        self.b2 = 0.0
    
    def get_pred(self):
        if self.b2 == 0:
            new_pred = 0.0
        else:
            new_pred = self.ab / self.b2
        return new_pred
    
    def set_outcome(self, a, b):
        self.ab += -a * b
        self.b2 += b * b

class NoMinimizer:
    def __init__(self, vmin=0.0, vmax=1.0, num_step=100):
        pass
    
    def get_pred(self):
        return 0.0
    
    def set_outcome(self, a, b):
        pass

    
minimizers = {'swap': SwapMinimizer, 'naive': NaiveMinimizer, 'avg': AvgMinimizer, 'none': NoMinimizer}
    


