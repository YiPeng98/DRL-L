import gym
import numpy as np
from gym import wrappers
from gym.envs.registration import register

def value_iteration(env, gamma = 1.0):
    v = np.zeros(env.env.nS)
    q = np.zeros((env.env.nS, env.env.nA))
    eps = 1e-10
    max_iterations = 100000
    for i in range(max_iterations):
        pre_v = np.copy(v)
        for s in range(env.env.nS):
            for a in range(env.env.nA):
                q[s][a] = sum([p*(r + gamma * pre_v[s_])  for p, s_, r, done in env.env.P[s][a]])
            v[s] = max(q[s])
        if(np.sum(np.fabs(v-pre_v)) <= eps):
            print('Value-iteration converged at iteration# %d.', (i+1))
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
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    optimal_value = value_iteration(env, gamma = 1.0)
    optimal_policy = extract_policy(optimal_value, gamma = 1.0)
    scores = evaluate_policy(env, optimal_policy, gamma = 1.0)
    print('Average scores = ', scores) 