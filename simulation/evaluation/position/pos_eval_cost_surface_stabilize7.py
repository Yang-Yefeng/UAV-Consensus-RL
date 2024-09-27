import os, sys, datetime, time, torch

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.envs.uav_pos_ctrl.uav_pos_stabilize_RL import uav_pos_stabilize_RL, uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from environment.envs.UAV.RobustDifferentatior_3rd import robust_differentiator_3rd as rd3
from algorithm.policy_base.Proximal_Policy_Optimization2 import Proximal_Policy_Optimization2 as PPO2
from utils.classes import *
from utils.functions import *

'''Parameter of the UAV'''
DT = 0.01
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
    # k1=np.array([4., 4., 15.]).astype(float),
    # k2=np.array([1., 1., 1.5]).astype(float),
    # k3=np.array([0.05, 0.05, 0.05]).astype(float),
    # k4=np.array([5, 4, 5]).astype(float),  # 要大
    k1=np.array([6.00810648, 6.80311651, 13.47563418]).astype(float),  # 手调: 4 4 15
    k2=np.array([2.04587905, 1.60844957, 0.98401018]).astype(float),  # 手调: 1 1 1.5
    k3=np.array([0.05, 0.05, 0.05]).astype(float),
    k4=np.array([9.85776965, 10.91725924, 13.90115023]).astype(float),  # 要大     手调: 5 4 5
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

pos_ctrl_param = fntsmc_param(
    k1=np.array([0.3, 0.3, 1.0]),
    k2=np.array([0.5, 0.5, 1]),
    k3=np.array([2, 2, 2]),  # 补偿观测器的，小点就行
    k4=np.array([3, 3, 3]),
    alpha1=np.array([1.01, 1.01, 1.01]),
    alpha2=np.array([1.01, 1.01, 1.01]),
    dim=3,
    dt=DT
)


def reset_pos_ctrl_param(flag: str):
    if flag == 'zero':
        pos_ctrl_param.k1 = 0.01 * np.ones(3)
        pos_ctrl_param.k2 = 0.01 * np.ones(3)
        pos_ctrl_param.k4 = 0.01 * np.ones(3)
    elif flag == 'random':
        pos_ctrl_param.k1 = np.random.random(3)
        pos_ctrl_param.k2 = np.random.random(3)
        pos_ctrl_param.k4 = np.random.random(3)
    else:  # optimal 手调的
        pos_ctrl_param.k1 = np.array([0.3, 0.3, 1.0])
        pos_ctrl_param.k2 = np.array([0.5, 0.5, 1])
        pos_ctrl_param.k4 = np.array([6, 6, 6])


if __name__ == '__main__':
    HEHE_FLAG = True
    
    env = uav_pos_stabilize_RL(uav_param, att_ctrl_param, pos_ctrl_param)
    env.Q_pos = np.array([1., 1., 1.])
    env.Q_vel = np.array([0.1, 0.1, 0.1])
    env.R = np.array([0.0, 0.0, 0.0])
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
    
    x_num = 11
    y_num = 11
    z_num = 11
    X = np.linspace(env.x_min + 1, env.x_max - 1, x_num)
    Y = np.linspace(env.y_min + 1, env.y_max - 1, y_num)
    Z = np.linspace(env.z_min + 1, env.z_max - 1, z_num)
    
    NN_c = 199
    save_path = os.path.dirname(os.path.abspath(__file__))  + '/test_cost_surface'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for i in range(10):
        i += 60
        print('Start: ', i + 1)
        opt_str = '/../../../datasave/log/stabilize_stage1/trainNum_' + str(10 * (i + 1)) + '/'
        opt_path = os.path.dirname(os.path.abspath(__file__)) + opt_str
        actor = PPOActor_Gaussian(state_dim=env.state_dim, action_dim=env.action_dim)
        actor.load_state_dict(torch.load(opt_path + 'actor', weights_only=True))
        agent = PPO2(env_msg=env_msg, ppo_msg=ppo_msg, actor=actor)
        
        env.load_norm_normalizer_from_file(opt_path, 'state_norm.csv')
        cost = np.zeros((x_num * y_num * z_num, 4))
        __i = 0
        for _x in X:
            for _y in Y:
                for _z in Z:
                    p = [[0, 0, 0, 0], [1, 1, 1, 1], [np.pi / 2, 0, 0, 0], [_x, _y, _z, 0]]     # 参考轨迹的参数，这里只用最后一组
                    reset_pos_ctrl_param('optimal')
                    obs = rd3(use_freq=True, omega=np.array([2, 2, 2]), dim=3, thresh=np.array([0.5, 0.5, 0.5]), dt=DT)
                    obs.reset()
                    env.reset_env(is_random=False, random_pos0=False, new_pos_ctrl_param=None, outer_param=p, is_ideal=False)
                    while not env.is_terminal:
                        _a = agent.evaluate(env.current_state_norm(env.current_state, update=False))
                        env.get_param_from_actor(_a, hehe_flag=HEHE_FLAG)  # 将控制器参数更新
                        
                        syst_dynamic = -env.kt / env.m * env.dot_eta() + env.A()
                        obs_eta, _ = obs.observe(x=env.eta(), syst_dynamic=syst_dynamic)
                        
                        dot_att_lim = [np.pi / 2, np.pi / 2, np.pi / 2]
                        att_lim = [np.pi / 3, np.pi / 3, np.pi]
                        action = env.generate_action_4_uav(att_lim=att_lim, dot_att_lim=dot_att_lim, obs=obs_eta)
                        env.step_update(action=action)
                        r = env.sum_reward
                        cost[__i][:] = np.array([_x, _y, _z, r])
                        __i += 1
        print('Finish: ', i + 1)
        pd.DataFrame(cost, columns=['x_d', 'y_d', 'z_d', 'r']).to_csv(save_path + '/cost_' + str(10 * (i + 1)) + '.csv', sep=',', index=False)
