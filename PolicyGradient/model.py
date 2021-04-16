# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    '''MLP
       input:state
       output:probability of action
    '''
    def __init__(self, state_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # 概率输出为1的原因是动作空间维度为2，所以此处设置输出为向左的概率，1-输出值为动作右的概率

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # 新版本中中torch.sigmoid
        return x