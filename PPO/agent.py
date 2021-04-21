# coding=utf-8

import os
import numpy as np
import torch
from torch import optim # 注意此处与GitHub中的不同
from PPO.model import Actor, Critic
from PPO.memory import PPOMemory

class PPO:
    def __init__(self, state_dim, action_dim, cfg):
        self.env = cfg.env
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.gae_lambda = cfg.gae_lambda
        self.n_epochs = cfg.n_epochs
        self.device = cfg.device
        self.loss = 0
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.critic = Critic(state_dim, cfg.hidden_dim).to(cfg.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), cfg.actor_lr)
        self.memory = PPOMemory(cfg.batch_size)

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        dist = self.actor(state) # 返回的是根据网络输出的动作概率的一个分布，用来抽样要执行的动作
        value = self.critic(state)
        action = dist.sample()
        ln_probs = dist.log_prob(action).item()
        value =  value.item()
        action = action.item()
        return action, ln_probs, value

    def update(self):
        for _ in range(self.n_epochs): # 本批数据的使用次数
            # 从经验池中提取数据
            state_arr, reward_arr, action_arr, dones_arr, old_prob_arr, vals_arr, batches = self.memory.sample()
            values = vals_arr

            # 计算Advantage
            # ''' 顺序计算Advantage'''
            # advantage1 = np.zeros(len(reward_arr), dtype=np.float32)
            # for t in range(len(reward_arr)-1): # 倒序可能会更加单，后期可以改一个倒叙版本,len(reward_arr)-1的原因是下边的k+1
            #     discount =1 
            #     a_t = 0
            #     for k in range(t, len(reward_arr)-1):
            #         a_t += discount*(reward_arr[k]+self.gamma*values[k+1]*(1-int(dones_arr[k]))-values[k])
            #         discount *= self.gamma*self.gae_lambda
            #     advantage1[t] = a_t
            # advantage1 = torch.tensor(advantage1).to(self.device)

            '''倒序版本计算Advantage'''
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            a_t = 0
            for t in reversed(range(len(reward_arr)-1)):
                a_t = reward_arr[t]+self.gamma*values[t+1]*(1-int(dones_arr[t]))-values[t] + self.gamma*self.gae_lambda*a_t
                advantage[t] = a_t               
            advantage = torch.tensor(advantage).to(self.device)
            
            # 梯度下降SGD
            values = torch.tensor(values).to(self.device)
            for batch in batches: # iteration过程，每个循环的一遍，一次参数更新。用minibatch时就意味着train完一个batch
                # 将经验池中提取的数据放到GPU
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                # 产生本批数据当前策略下的动作概率分布，并计算actor_loss 即actor的目标函数
                dist = self.actor(states) # 需要测试一下此处的dist与new_probs.exp()的值是否一致
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio # 此处采用ppo-clip算法
                weighted_clipped_probs = advantage[batch] * torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() # 计算批大小为5的数据的损失函数

                # 产生当前critic下的价值，并计算critic_loss
                critic_value = self.critic(states) # 计算出当前critic的价值
                returns = advantage[batch] + values[batch] # 按照dueling q-learning中优势函数加上状态价值函数得到动作价值函数Q
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear() # 该批数据利用完后清除

    def save(self, path):
        actor_checkpoint = os.path.join(path, self.env+'_actor.pt')
        critic_checkpoint= os.path.join(path, self.env+'_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)

    def load(self, path):
        actor_checkpoint = os.path.join(path, self.env+'_actor.pt')
        critic_checkpoint= os.path.join(path, self.env+'_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint)) 
        self.critic.load_state_dict(torch.load(critic_checkpoint))