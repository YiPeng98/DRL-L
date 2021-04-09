import gym
import numpy as np
env = gym.make('MountainCar-v0')
env.seed(0)

class BespokeAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, **args):
        pass

def play(env, agent, render=False, train=True):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward

        if train:
            agent.learn()
        
        if done:
            break
        observation = next_observation

    return episode_reward

if __name__ == "__main__":
    agent = BespokeAgent(env)
    episode_reward = [play(env, agent, True) for _ in range(10)]
    print('平均回合奖励 = {}'.format(np.mean(episode_reward)))
    env.close() # 交互一个回合看看
