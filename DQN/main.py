import sys, os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

SAVED_MODEL_PATH = curr_path + '/save_model/' 
RESULT_PATH = curr_path + '/results/'

''' 以下创建目录的方式不推荐，
    在os.path.exists()和os.makedirs()之间的时间
    可能会出现目录被创建
'''
# if not os.path.exists(SAVED_MODEL_PATH):
#     os.mkdir(SAVED_MODEL_PATH)
# if not os.path.exists(RESULT_PATH:
#     os.mkdir(RESULT_PATH)

import gym
import torch
from DQN.agent import DQN
from common.plot import plot_rewards
from common.utils import save_results, mk_dir, del_empty_dir

mk_dir(SAVED_MODEL_PATH, RESULT_PATH)
del_empty_dir(SAVED_MODEL_PATH, RESULT_PATH)

class DQNConfig:
    def __init__(self):
        self.env = 'CartPole-v0'
        self.algo = 'DQN'
        self.gamma = 0.95
        self.epsilon_start = 1
        self.epsilon_end = 0.01
        self.epsilon_decay = 100
        self.lr = 0.0001
        self.memory_capacity = 100000
        self.batch_size = 64
        self.train_eps = 300
        self.train_steps = 1000
        self.target_update = 2
        self.eval_eps = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = 256
        self.result_path = curr_path+"/results/" +self.env+'/'

def train(cfg, env, agent):
    print('Start to train !')
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done) # 此处存入经验池中的数据是np类型的，不能传播梯度等
            state = next_state
            agent.update() # 此处没有了之前的判断语句，因为判断语句写在了update()函数里边
        
        # 更新目标网络
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:{}/{},Reward:{}'.format(i_episode+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training')
    return rewards,ma_rewards

if __name__ == "__main__":
    cfg = DQNConfig()
    env = gym.make(cfg.env)
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim,action_dim,cfg)
    rewards,ma_rewards = train(cfg,env,agent)
    make_dir(cfg.result_path)
    agent.save(path=cfg.result_path)
    save_results(rewards,ma_rewards,tag='train',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="train",algo=cfg.algo,path=cfg.result_path)