# coding=utf-8

import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # 此处的目的是在buffer列表还没有满的时候，向其中添加元素，一般与可以直接用position进行索引
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch) # 将batch中的数据按照对应关系压缩为元组
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)