#!/usr/bin/env python
# conding=utf-8
'''
以下是将终端路径添加到系统路径中内容
'''
import sys
import os
curr_path = os.path.dirname(__file__)
# __file__就是当前脚本运行的路径 e:/Work/Python/DRL-L/Qlearning/main.py ，dirname获取当前目录的父目录
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path) # add current terminal path to sys.path 是为了能够正确导入同级目录下的包

import gym
import datetime
from envs.gridworld_env import CliffWalkingWapper, FrozenLakeWapper