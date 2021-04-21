# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from common.memory import ReplayBuffer
from common.model import MLP

class DQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.loss = 0
        self.gamma = cfg.gamma
        self.frame_idx = 0 # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.policy_net = MLP(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_net = MLP(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state):
        '''policy_net负责与环境进行互动并产生相关动作存放到经验池中，因为后边会采样经验池中的数据来重新生成相关Q值，所以此处不进行梯度的更新'''
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad(): # 使用该语句，使policy_net网络不会进行更新
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_value = self.policy_net(state) 
                action = q_value.max(1)[1].item() # tensor.max(1)[1]返回最大值对应的下标，即action
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        ''' 转换为Tensor
        '''
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        # 计算当前(s_t, a)对应的Q值,此处的Q值用来训练，所以要求计算梯度;
        # 其实也可以在choose_action时将q值存到经验池中，就可以不同进行下一步的计算了
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)

        # 计算s_t+1状态下target_net网络的最大值
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach() # 由target_net输出的值不会参与到梯度的计算中
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        self.loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'DQN_CheckPoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'DQN_CheckPoint.pth'))