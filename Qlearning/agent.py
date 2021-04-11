# coding=utf-8

import numpy as np
import math
import torch
from collections import defaultdict

class QLearning(object):
    def __init__(self, action_dim, cfg):
        self.action_dim = action_dim
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon_decay = cfg.epsilon_decay
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon = 0
        self.sample_count = 0
        self.Q_table = defaultdict(lambda: np.zeros(action_dim)) # A nested dictionary that maps state -> (action -> action-value)

    def choose_action(self, state): # 此处是执行策略，采用e-greedy策略
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-(self.sample_count / self.epsilon_decay))
        # e-greedy policy
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.action_dim)
        return int(action)    

    def update(self, state, action, reward, next_state, done): # 执行策略和学习策略不同
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] = Q_predict + self.lr * (Q_target - Q_predict)

    def save(self, path):
        import dill
        torch.save(obj=self.Q_table, f=path + "Qlearning_model.pkl", pickle_module=dill) # dill是将python对象以字节流的形式存储在文件中

    def load(self, path):
        import dill
        self.Q_table = torch.load(f=path+"Qlearning_model.pkl", pickle=dill)