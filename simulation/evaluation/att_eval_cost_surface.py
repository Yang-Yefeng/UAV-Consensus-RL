import os, sys, datetime, time, torch

import numpy as np
import pandas as pd

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

    HEHE_FLAG = True

    opt_path = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/att_maybe_good_1/'

    env = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    env.load_norm_normalizer_from_file(opt_path, 'state_norm.csv')

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
    
    A_num = 70
    T_num = 15
    A = np.linspace(0, deg2rad(60), A_num)
    T = np.linspace(3,6,T_num)
    _i = 0
    
    cost = np.zeros((A_num * T_num, 4))  # a, t, r1 ,r2
    for a in A:
        for t in T:
            p = np.array([[a, a, a], [t, t, t], [0, 0, 0]])
            '''1. RL'''
            reset_att_ctrl_param('optimal')
            env.reset_env(random_att_trajectory=True, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param, outer_param=p)
            while not env.is_terminal:
                _a = agent.evaluate(env.current_state_norm(env.current_state, update=False))
                env.get_param_from_actor(_a, hehe_flag=HEHE_FLAG)  # 将控制器参数更新
                _rhod = env.rho_d_all[env.n]
                _dot_rhod = env.dot_rho_d_all[env.n]
                _dot2_rhod = env.dot2_rho_d_all[env.n]
                _torque = env.att_control(_rhod, _dot_rhod, _dot2_rhod, True)
                env.step_update([_torque[0], _torque[1], _torque[2]])
            r1 = env.sum_reward
            
            '''2. 传统'''
            reset_att_ctrl_param('optimal')
            env.reset_env(random_att_trajectory=True, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param, outer_param=p)
            while not env.is_terminal:
                _rhod = env.rho_d_all[env.n]
                _dot_rhod = env.dot_rho_d_all[env.n]
                _dot2_rhod = env.dot2_rho_d_all[env.n]
                _torque = env.att_control(_rhod, _dot_rhod, _dot2_rhod, True)
                env.step_update([_torque[0], _torque[1], _torque[2]])
            r2 = env.sum_reward
            cost[_i, :] = np.array([a, t, r1, r2])
            _i += 1
            if _i % 50 == 0:
                print(_i)
            # env.visualization()
    
    print('Finish...')
    pd.DataFrame(cost, columns=['A', 'T', 'r1', 'r2']).to_csv('att_cost_surface.csv', sep=',', index=False)
