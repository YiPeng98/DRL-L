# coding=utf-8

import torch
import torch.nn as nn
import numpy as np
import random, math
import torch.optim as optim
from common.model import MLP
from common.memory import ReplayBuffer

class HierarchicalDQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.policy_net = MLP(2*state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.meta_policy_net = MLP(state_dim, state_dim, cfg.hidden_dim).to(cfg.device) # 高层策略用于产生高层指导动作，输出动作分布等价于状态分布
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.meta_optimizer = optim.Adam(self.meta_policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.meta_memory = ReplayBuffer(cfg.memory_capacity)
        self.loss_numpy = 0
        self.meta_loss_numpy = 0
        self.losses = []
        self.meta_losses = []

    def set_goal(self, state): # 从状态维度范围中采取一个值作为高层策略，也是采用Q-learning方法
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad(): # 在产生数据时候没有比较进行训练
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                goal = self.meta_policy_net(state).max(1)[1].item() # 为什么此处要抽取一个最大值作为高层策略呢，理由是算法是H-DQN
        else:
            goal = random.randrange(self.state_dim)
        return goal

    def to_onehot(self, x): # 将goal转换为one-hot向量便于与state进行拼接
        oh = np.zeros(self.state_dim)
        oh[x-1] = 1
        return oh

    def choose_action(self, state): # 输入的是one-hot 和 state的拼接
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_value = self.policy_net(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def update_policy(self):
        if self.batch_size > len(self.memory):
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(np.float32(done_batch)).to(self.device)

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch).squeeze(1)
        next_state_values = self.policy_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.loss_numpy = loss.detach().cpu().numpy()
        self.losses.append(self.loss_numpy)

    def update_meta(self):
        if self.batch_size > len(self.meta_memory):
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.meta_memory.sample(self.batch_size)
        
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(np.float32(done_batch)).to(self.device)

        q_values = self.meta_policy_net(state_batch).gather(dim=1, index=action_batch).squeeze(1)
        next_state_values = self.meta_policy_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        meta_loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        meta_loss.backward()
        for param in self.meta_policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.meta_optimizer.step()
        self.meta_loss_numpy = meta_loss.detach().cpu().numpy()
        self.meta_losses.append(self.meta_loss_numpy)

    def update(self):
        self.update_policy()
        self.update_meta()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path+'policy_checkpoint.pth')
        torch.save(self.meta_policy_net.state_dict(), path+'meta_checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path+'policy_checkpoint.pth'))
        self.meta_policy_net.load_state_dict(torch.load(path+'meta_checkpoint.pth'))