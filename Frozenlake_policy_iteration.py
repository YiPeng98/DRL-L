import gym
import numpy as np
from gym import wrappers
from gym.envs.registration import register

def compute_policy_v(env, policy, gamma):
    '''利用贝尔曼方程进行迭代计算policy的价值'''
    #初始化状态价值函数
    v = np.zeros(env.env.nS)
    eps = 1e-10
    while True:
        pre_v = np.copy(v)
        for s in range(env.env.nS):
            #根据当前给定的策略来输出每个状态下的动作值，即有了s_t和a_t
            policy_a = policy[s] 
            #迭代更新状态价值函数:根据s_t和a_t计算出下一状态、转移概率、及时奖励是否完成游戏
            v[s] = sum([p*(r + gamma*(pre_v[s_]))  for p, s_, r, done in env.env.P[s][policy_a]]) 
        if(np.sum(np.fabs(v-pre_v)) <= eps):
            # 迭代以完成
            break
    return v

def extract_policy(v, gamma):
    '''在给定v的条件下提取出当前最佳策略'''
    policy = np.zeros(env.env.nS)
    q = np.zeros((env.env.nS, env.env.nA))
    for s in range(env.env.nS):
        for a in range(env.env.nA):
            q[s][a] = sum([p*(r + gamma * v[s_])  for p, s_, r, done in env.env.P[s][a]])
    policy =  np.argmax(q, axis=1)
    return policy

def policy_iteration(env, gamma):
    policy = np.random.choice(env.env.nA, size=(env.env.nS)) #初始化策略，即给4X4个网格从动作空间中挑选一个动作
    max_iterations = 20000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if(np.all(policy == new_policy)): 
            #策略迭代已完成
            print('Policy-Iteration converged at step %d.' %(i + 1))
            break
        policy = new_policy
    return policy

def run_episode(env, policy, gamma = 1.0, render= False):
    '''运行一回合并计算该回合的reward'''
    o = env.reset()
    episode_r = 0
    step_i = 0
    while True:
        if render:
            env.render()
        a = policy[o]
        o, r, done, _ = env.step(a)
        episode_r += (gamma ** step_i * r) #此处使用带有折扣因子的回报计算方式，考虑到对于未来达到的状态是未知的，所以要进行折扣
        step_i += 1
        if(done):
            break
    return episode_r

def evaluate_policy(env, policy, gamma, n=100):
    '''执行100回合当前策略，并且计算出平均回报'''
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

if __name__ == '__main__':
    env_name = 'FrozenLake-v0' # 环境名称
    env = gym.make(env_name)
    env.reset()
    optimal_policy = policy_iteration(env, gamma=1.0)
    scores = evaluate_policy(env, optimal_policy, gamma=1.0)
    print('Average scores = ', scores)

