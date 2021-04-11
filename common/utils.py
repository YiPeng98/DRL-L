# coding=utf-8

import os
import numpy as np

def save_results(rewards, ma_rewards, tag='train', path='./results'):
    np.save(path+'rewards_'+tag+'.npy', rewards) # .npy文件是NumPy库存储数据的文件
    np.save(path+'ma_rewards_'+tag+'.npy', ma_rewards)
    print('results saved!')