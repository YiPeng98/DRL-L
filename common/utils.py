# coding=utf-8

import os
import numpy as np
from pathlib import Path

def save_results(rewards, ma_rewards, tag='train', path='./results'):
    np.save(path+'rewards_'+tag+'.npy', rewards) # .npy文件是NumPy库存储数据的文件
    np.save(path+'ma_rewards_'+tag+'.npy', ma_rewards)
    print('results saved!')

def mk_dir(*paths): # 一个*表示将参数以元组的形式传入；两个*表示将参数以字典的形式传入
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def del_empty_dir(*paths):
    '''目的是删除paths下的所有空文件夹
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path+dir)):
                os.removedirs(os.path.join(path+dir))
        