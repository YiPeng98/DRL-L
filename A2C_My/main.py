# coding=utf-8

import os,sys
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import torch
import gym
import datetime
from common.utils import save_results, mk_dir, del_empty_dir
from common.plot import plot_rewards
from agent import A2C

# 创建相关文件夹
SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = curr_path + '/save_model/'
RESULT_PATH = curr_path + '/results/' + SEQUENCE + '/'
# 推荐使用以下创建目录的方式
mk_dir(SAVED_MODEL_PATH, RESULT_PATH)
del_empty_dir(RESULT_PATH)

class A2CConfig:
    def __init__(self):
        self.env = 'CartPole-v0'
        self.algo = 'A2C'
        self.gamma = 0.99
        self.lr = 3e-4
        self.train_eps = 200
        self.train_steps = 200
        self.hidden_dim = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = SAVED_MODEL_PATH

def train(cfg, env, agent):
    print('Start to train!')
    rewards = []
    ma_rewards = []
    # 首先判断是否已经有模型存在，存在就加载上去
    if os.listdir(cfg.model_path):
        agent.load(SAVED_MODEL_PATH)
    for i_episode in range(cfg.train_eps):
        #  定义每一回合内部用到的变量
        done = False
        state = env.reset()
        ep_step = 0
        entropy = 0
        values = []
        next_values = []
        log_probs = []
        step_rewards = []
        mask_dones = []
        while not done: # 先产生一回合的数据
            action, value, dist = agent.choose_action(state)
            # 得到与环境互动一步后的数据
            next_state, reward, done, _ = env.step(action) # 没有往经验池中存储数据
            # 将相关数据添加到指定列表
            _, next_value, _ = agent.choose_action(next_state)
            log_prob = dist.log_prob(torch.tensor(action).to(cfg.device))

            log_probs.append(log_prob)
            values.append(value)
            next_values.append(next_value)
            step_rewards.append(torch.FloatTensor([reward]).to(cfg.device))
            mask_dones.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))

            ep_step += 1
            entropy += dist.entropy().mean()
            state = next_state
        mask_dones[-1] = torch.FloatTensor([0.0]).unsqueeze(1).to(cfg.device)
        ep_reward = sum(step_rewards)
        print('Episode:{}/{}, Reward:{}, Steps:{}'.format(i_episode+1, cfg.train_eps, ep_reward.item(), ep_step))
        # 利用一回合的数据进行更新
        agent.update(values, next_values, step_rewards, log_probs, mask_dones, entropy)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    # 保存训练好的模型
    print('Complete one_eps training!')
    agent.save(SAVED_MODEL_PATH)
    return rewards, ma_rewards


if __name__ == '__main__':
    cfg = A2CConfig()
    env = gym.make(cfg.env)
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2C(state_dim, action_dim, cfg)
    rewards, ma_rewards =  train(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='train', path=RESULT_PATH)
    plot_rewards(rewards, ma_rewards, tag="train", algo = cfg.algo, path=RESULT_PATH)