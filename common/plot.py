# coding=utf-8

import matplotlib.pyplot as plt
import seaborn as sns

def plot_rewards(rewards, ma_rewards, tag="train", algo="DQN", save=True, path='./'):
    sns.set() # 对seaborn库默认的组合进行调用
    plt.title("average learning curve of {}".format(algo))
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards') # 画图
    plt.plot(ma_rewards, label='moving average rewards') # 画图
    plt.legend() # 给图像的一角加图例
    if save:
        plt.savefig(path+"rewards_curve_{}".format(tag))
    plt.show() # 显示已经画好的图

# def plot_losses(losses, algo="DQN", save=True, path='./'):
