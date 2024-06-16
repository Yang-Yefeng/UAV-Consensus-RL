import os, sys, datetime, time, torch

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.envs.uav_pos_ctrl.uav_pos_stabilize_RL import uav_pos_stabilize_RL, uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from algorithm.policy_base.Proximal_Policy_Optimization2 import Proximal_Policy_Optimization2 as PPO2
from utils.classes import *
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

pos_ctrl_param = fntsmc_param(
    k1=np.array([0.3, 0.3, 1.0]),
    k2=np.array([0.5, 0.5, 1]),
    k3=np.array([0.05, 0.05, 0.05]),  # 补偿观测器的，小点就行
    k4=np.array([6, 6, 6]),
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
    
    opt_path = os.path.dirname(os.path.abspath(__file__)) + '/../../../datasave/nets/pos_maybe_good_3/'
    
    env = uav_pos_stabilize_RL(uav_param, att_ctrl_param, pos_ctrl_param)
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
    
    x_num = 10
    y_num = 10
    x = np.linspace(env.x_min + 2, env.x_max - 2, x_num)
    y = np.linspace(env.y_min + 2, env.y_max - 2, y_num)
    z = 3.0
    _i = 0
    
    cost = np.zeros((x_num * y_num, 4))  # a, t, r1 ,r2
    for _x in x:
        for _y in y:
            p = [[_x, _y, z]]
            '''1. RL'''
            reset_pos_ctrl_param('optimal')
            env.reset_env(is_random=False,random_pos0=False, new_pos_ctrl_param=None, outer_param=p)
            while not env.is_terminal:
                _a = agent.evaluate(env.current_state_norm(env.current_state, update=False))
                env.get_param_from_actor(_a, hehe_flag=HEHE_FLAG)  # 将控制器参数更新
                
                dot_att_lim = [np.pi / 2, np.pi / 2, np.pi / 2]
                action = env.generate_action_4_uav(att_lim=[np.pi / 3, np.pi / 3, np.pi], dot_att_lim=dot_att_lim)
                env.step_update(action=action)
            r1 = env.sum_reward
            
            '''2. 传统'''
            reset_pos_ctrl_param('optimal')
            env.reset_env(is_random=False,random_pos0=False, new_pos_ctrl_param=None, outer_param=p)
            while not env.is_terminal:
                dot_att_lim = [np.pi / 2, np.pi / 2, np.pi / 2]
                action = env.generate_action_4_uav(att_lim=[np.pi / 3, np.pi / 3, np.pi], dot_att_lim=dot_att_lim)
                env.step_update(action=action)
            r2 = env.sum_reward
            cost[_i, :] = np.array([_x, _y, r1, r2])
            _i += 1
            if _i % 50 == 0:
                print(_i)
            print('r1:', r1, 'r2', r2)
            # env.visualization()
    
    print('Finish...')
    save_path = env.project_path + '/plot/3D_surface/position/'
    pd.DataFrame(cost, columns=['A', 'T', 'r1', 'r2']).to_csv(save_path + 'pos_stabilize_cost_surface.csv', sep=',', index=False)
