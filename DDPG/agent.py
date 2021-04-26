# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.model import Actor, Critic
from common.memory import ReplayBuffer

class DDPG:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        # 定义网络
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        # 定义优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau # 目标网络软更新
        self.gamma = cfg.gamma
        # 同步target网络
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0, 0] # 经验池中的数据不能进行梯度的反向传播

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 从经验池中抽样出数据
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # 将所有变量转换为张量
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device) # done为np类型的数据，因此使用np.float32进行类型转换
        
        # 计算价值损失
        value = self.critic(state, action) # 此处动作直接从经验池中拿去是因为此时只是更新critic网络
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())# 因为target_net不进行参数更新
        expected_value = reward + target_value * self.gamma * (1.0 - done)
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
        value_loss = nn.MSELoss()(value, expected_value.detach()) #只要截断此处就可以不会对target_net进行参数更新了
        
        # 计算策略损失
        policy_loss = -self.critic(state, self.actor(state)) # 此处动作要用actor来产生，原因是用经验池子中的数据将不会更新actor，还有一点原因就是经验池子中的数据因为加了噪声的原因，已经不是策略网络产生的数据了

        # 更新策略网络
        self.actor_optimizer.zero_grad()
        policy_loss.backward(torch.ones_like(policy_loss)) # 当policy为张量时，需要输入参数torch.ones_like(policy_loss)
        self.actor_optimizer.step()

        # 更新critic网络
        self.critic_optimizer.zero_grad()
        value_loss.backward(torch.ones_like(value_loss))
        self.critic_optimizer.step()
        
        # 软更新更新目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def save(self, path):
        torch.save(self.actor.state_dict(), path+'checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path+'checkpoint.pt'))