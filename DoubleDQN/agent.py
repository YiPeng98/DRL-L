# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import math
from common.model import MLP
from common.memory import ReplayBuffer
import random

class DoubleDQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = cfg.gamma
        self.policy_net = MLP(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_net = MLP(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # 不启用 BatchNormalization 和 Dropout
        self.optim = optim.Adam(self.policy_net.parameters(), lr = cfg.lr)
        self.device = cfg.device
        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.loss = 0

    def choose_action(self, state):
        self.frame_idx += 1
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad(): # 此处不进行梯度传播
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # 抽样数据
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        # 将数据转换为Tensor并推送到GPU
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, device=self.device)

        # 产生(s_t,a)下的q值
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)

        # 计算next_q_values
        next_action_batch = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1) # 此处就是DoubleDQN的关键，动作的选取是通过policy_net的
        next_q_values = self.target_net(next_state_batch).gather(dim=1, index=next_action_batch).detach().squeeze(1) # q值是target_net输出的
        expected_q_values = reward_batch + self.gamma * next_q_values * (~done_batch)

        self.loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.optim.zero_grad()
        self.loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'DQN_CheckPoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'DQN_CheckPoint.pth'))