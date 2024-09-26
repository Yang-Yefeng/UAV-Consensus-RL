import sys, os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.envs.UAV.FNTSMC import fntsmc_param
from environment.envs.uav_pos_ctrl.uav_pos_stabilize_RL import uav_pos_stabilize_RL, uav_param
from utils.functions import *


'''Parameter of the UAV'''
DT = 0.02
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 10
'''Parameter of the UAV'''


'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param(
    k1=np.array([6.00810648, 6.80311651, 13.47563418]).astype(float),			# 手调: 4 4 15
    k2=np.array([2.04587905, 1.60844957, 0.98401018]).astype(float),			# 手调: 1 1 1.5
    k3=np.array([0.05, 0.05, 0.05]).astype(float),
    k4=np.array([9.85776965, 10.91725924, 13.90115023]).astype(float),       # 要大     手调: 5 4 5
    alpha1=np.array([1.01, 1.01, 1.01]).astype(float),
    alpha2=np.array([1.01, 1.01, 1.01]).astype(float),
    dim=3,
    dt=DT
    # k1 控制滑模中 e 的占比，k3 控制滑模中 sig(e)^alpha1 的占比
    # k2-alpha1 的组合不要太大
    # k3 是观测器补偿用的，实际上观测的都很好，所以 k4 要大于0，但是非常小
    # k4-alpha2 的组合要比k3-alpha1 大，但是也别太大
)
'''Parameter list of the attitude controller'''


'''Parameter list of the position controller'''
pos_ctrl_param = fntsmc_param(
    k1=np.array([0.6, 0.6, 1.0]),
    k2=np.array([0.5, 0.5, 1]),
    k3=np.array([2, 2, 2]),        # 补偿观测器的，小点就行
    k4=np.array([3, 3, 3]),
    alpha1=np.array([1.01, 1.01, 1.01]),
    alpha2=np.array([1.01, 1.01, 1.01]),
    dim=3,
    dt=DT
)
'''Parameter list of the position controller'''

if __name__ == '__main__':
	env = uav_pos_stabilize_RL(uav_param, att_ctrl_param, pos_ctrl_param)
	NUM_OF_SIMULATION = 5
	cnt = 0
	success = 0
	while cnt < NUM_OF_SIMULATION:
		'''1. reset and generate reference signal'''
		p = [[2, 2, 2, deg2rad(70)], [5, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0]]
		env.reset_env(is_random=True, random_pos0=True, new_pos_ctrl_param=None, outer_param=None)
		if cnt % 1 == 0:
			print('Current:', cnt)
		
		'''2. control'''
		while not env.is_terminal:
			dot_att_lim = [np.pi / 2, np.pi / 2, np.pi / 2]
			action = env.generate_action_4_uav(att_lim=[np.pi / 3, np.pi / 3, np.pi], dot_att_lim=dot_att_lim)
			env.step_update(action=action)
			
			env.image = env.image_copy.copy()
			env.visualization()
		if env.terminal_flag == 1:
			success += 1
		print(env.sum_reward)
		cnt += 1
	print('success rate: ', success / NUM_OF_SIMULATION)
