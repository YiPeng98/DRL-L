# coding=utf-8

import torch
from model import ActorCritic
import torch.optim as optim
import torch.nn as nn
from numpy import *
import os

class A2C:
    def __init__(self, state_dim, action_dim, cfg):
        self.gamma = cfg.gamma
        self.model = ActorCritic(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.device = cfg.device
        self.loss = 0
        self.env = cfg.env

    def choose_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        dist, value = self.model(state)
        action = dist.sample().item()
        return action, value, dist

    def update(self, values, next_values, step_rewards, log_probs, mask_dones, entropy): # 利用一回合数据进行更新
        expected_values = []
        advantages = []
        actor_losses = []
        critic_losses = []
        for step in range(len(step_rewards)):
            expected_values.append(step_rewards[step].item() + self.gamma * next_values[step].squeeze().item() * mask_dones[step].squeeze().item()) 
            advantages.append(expected_values[step] - values[step].item())
            actor_losses.append(-advantages[step] * log_probs[step].item())
            critic_losses.append(nn.MSELoss()(values[step].squeeze(), torch.tensor([expected_values[step]]).to(self.device)).cpu().detach().numpy())
        actor_loss = mean(actor_losses)
        critic_loss = mean(critic_losses)
        self.loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def save(self, path):
        model_checkpoint = os.path.join(path, self.env+'actor_critic.pt')
        torch.save(self.model.state_dict(), model_checkpoint)
        print('Model Saved!')

    def load(self, path):
        model_checkpoint = os.path.join(path, self.env+'actor_critic.pt')
        self.model.load_state_dict(torch.load(model_checkpoint))
        print('Model Loaded!')