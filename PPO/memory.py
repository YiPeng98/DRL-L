# coding=utf-8

import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.ln_probs = []
        self.vals = []
        self.batch_size = batch_size

    # sample函数的作用返回经验池中的数据，并返回随机划分好的batches
    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size) # 形如：[0, 5, 10, 15]
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices) # 随机化处理
        batches = [indices[i:i+self.batch_size] for i in batch_step] # 将indices按照batch_step进行划分为4批数据
        return np.array(self.states),\
                np.array(self.rewards),\
                np.array(self.actions),\
                np.array(self.dones),\
                np.array(self.ln_probs),\
                np.array(self.vals),\
                batches

    def push(self, state, reward, action, done, ln_prob, val):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.dones.append(done)
        self.ln_probs.append(ln_prob)
        self.vals.append(val)

    def clear(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.ln_probs = []
        self.vals = []