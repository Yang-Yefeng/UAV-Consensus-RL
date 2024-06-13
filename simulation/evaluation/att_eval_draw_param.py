import os, sys, datetime, time, torch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.envs.uav_att_ctrl.uav_att_ctrl_RL import uav_att_ctrl_RL, uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param

from algorithm.policy_base.Proximal_Policy_Optimization2 import Proximal_Policy_Optimization2 as PPO2

from utils.classes import *
from utils.functions import *

timestep = 0
ENV = 'uav_att_ctrl_RL'
ALGORITHM = 'PPO2'

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
    k1=np.array([4., 4., 15.]).astype(float),
    k2=np.array([1., 1., 1.5]).astype(float),
    k3=np.array([0.05, 0.05, 0.05]).astype(float),
    k4=np.array([5, 4, 5]).astype(float),  # 要大
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


def reset_att_ctrl_param(flag: str):
    if flag == 'zero':
        att_ctrl_param.k1 = 0.01 * np.ones(3)
        att_ctrl_param.k2 = 0.01 * np.ones(3)
        att_ctrl_param.k4 = 0.01 * np.ones(3)
    elif flag == 'random':
        att_ctrl_param.k1 = np.random.random(3)
        att_ctrl_param.k2 = np.random.random(3)
        att_ctrl_param.k4 = np.random.random(3)
    else:  # optimal 手调的
        att_ctrl_param.k1 = np.array([4., 4., 15.])
        att_ctrl_param.k2 = np.array([1., 1., 1.5])
        att_ctrl_param.k4 = np.array([5, 4, 5])


if __name__ == '__main__':
    # log_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/log/'
    # if not os.path.exists(log_dir):
    # 	os.makedirs(log_dir)
    # simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    # os.mkdir(simulationPath)

    opt_path = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/att_maybe_good_1/'        # att_maybe_good_1 目前位置最好的
    # opt_path = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/att_good_2/'  # att_good_2
    # opt_path = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/log/att_train_4/trainNum_5200/'

    env = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    env.load_norm_normalizer_from_file(opt_path, 'state_norm.csv')
    
    HEHE_FLAG = True
    if HEHE_FLAG:
        hehe = np.array([5, 5, 5, 1, 1, 1, 5, 5, 5])
    else:
        hehe = np.ones(env.action_dim)
    
    env_msg = {'state_dim': env.state_dim, 'action_dim': env.action_dim, 'name': env.name, 'action_range': env.action_range}
    ppo_msg = {'gamma': 0.99,
               'K_epochs': 10,
               'eps_clip': 0.2,
               'buffer_size': int(env.time_max / env.dt) * 2,
               'state_dim': env.state_dim,
               'action_dim': env.action_dim,
               'a_lr': 1e-5,
               'c_lr': 1e-4,
               'set_adam_eps': True,
               'lmd': 0.95,
               'use_adv_norm': True,
               'mini_batch_size': 64,
               'entropy_coef': 0.01,
               'use_grad_clip': True,
               'use_lr_decay': True,
               'max_train_steps': int(5e6),
               'using_mini_batch': False}

    actor = PPOActor_Gaussian(state_dim=env.state_dim, action_dim=env.action_dim)
    actor.load_state_dict(torch.load(opt_path + 'actor'))
    agent = PPO2(env_msg=env_msg, ppo_msg=ppo_msg, actor=actor)
    opt_param = np.zeros((int(uav_param.time_max / DT), env.action_dim))

    N = 5
    cal_param_mean = np.zeros((N, env.action_dim))
    for i in range(N):
        reset_att_ctrl_param('zero')
        env.reset_env(random_att_trajectory=True, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)
        while not env.is_terminal:
            _a = agent.evaluate(env.current_state_norm(env.current_state, update=False))
            env.get_param_from_actor(_a, hehe_flag=HEHE_FLAG)  # 将控制器参数更新
            opt_param[env.n] = _a * hehe
            _rhod = env.rho_d_all[env.n]
            _dot_rhod = env.dot_rho_d_all[env.n]
            _dot2_rhod = env.dot2_rho_d_all[env.n]
            _torque = env.att_control(_rhod, _dot_rhod, _dot2_rhod, True)
            env.step_update([_torque[0], _torque[1], _torque[2]])
            
            env.visualization()

        print('Evaluating %.0f | Reward: %.2f ' % (i, env.sum_reward))
        row = opt_param.shape[0]
        p = opt_param[int(0.75 * row): row, :]
        cal_param_mean[i, :] = np.mean(p, axis=0)
        
        time = np.linspace(0, env.time_max, int(uav_param.time_max / DT))
        plt.figure(0)
        plt.plot(time, opt_param[:, 0])  # k1 x
        plt.plot(time, opt_param[:, 1])  # k1 y
        plt.plot(time, opt_param[:, 2])  # k1 z
        plt.grid(True)

        plt.figure(1)
        plt.plot(time, opt_param[:, 3])  # k2 x
        plt.plot(time, opt_param[:, 4])  # k2 y
        plt.plot(time, opt_param[:, 5])  # k2 z
        plt.grid(True)

        plt.figure(2)
        plt.plot(time, opt_param[:, 6])  # k4 x
        plt.plot(time, opt_param[:, 7])  # k4 y
        plt.plot(time, opt_param[:, 8])  # k4 z
        plt.grid(True)

        plt.show()
    print(cal_param_mean)
    print(np.mean(cal_param_mean, axis=0))
