# coding=utf-8

import torch
from torch.distributions import Bernoulli # torch.distributions提供常见的概率分布
from torch.autograd import Variable
import numpy as np
from PolicyGradient.model import MLP

class PolicyGradient:
    def __init__(self, state_dim, cfg):
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, cfg.hidden_dim)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr)

    def choose_action(self, state): # state参数是numpy类型的[1,4]列表
        state = torch.from_numpy(state).float() # 变成tensor类型
        state = Variable(state) # 只有模型参数require_grad才要设置为True，此处叶子节点state不需要梯度更新
        probs = self.policy_net(state)
        m = Bernoulli(probs) # 创建参数化的伯努利分布：以probs的概率选择1，1-probs的概率选择0
        action = m.sample() # 通过伯努利分布采样动作
        action = action.data.numpy().astype(int)[0]
        return action

    def update(self, reward_pool, state_pool, action_pool):
        # 折扣因子奖励计算
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0: # 此处与main.py第52行将reward值设置为0有关，因为在reward中没有划分episode，只是将batch_size个episode的值班全部存入其中，因此通过设置终止状态的reward来划分不同的episode
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add # 此处对于缓冲池进行了再利用，从原来存储每一步的reward值变为了存储当前step的return

        ''' 标准化奖励，使之为正太分布，是训练技巧，算法中并未提及'''
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # 梯度下降
        self.optimizer.zero_grad()
        for i in range(len(reward_pool)):
            # 提取轨迹中的数据
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]

            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            s = m.log_prob(action)
            loss = -m.log_prob(action) * reward  # 在此处就是根据传入的动作值选择相应概率进行自然对数的计算
            loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path+'pg_checkpoint.pt')

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path+'pg_checkpoint.pt'))
