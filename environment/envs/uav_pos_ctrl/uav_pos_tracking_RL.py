import sys, os

from environment.envs.UAV.uav import uav_param
from environment.envs.UAV.uav_pos_ctrl import uav_pos_ctrl, fntsmc_param
from environment.color import Color
from environment.envs.UAV.ref_cmd import *

from algorithm.rl_base.rl_base import rl_base

from utils.classes import Normalization
from utils.functions import *

import cv2 as cv
import pandas as pd


class uav_pos_tracking_RL(rl_base, uav_pos_ctrl):
    def __init__(self, _uav_param: uav_param, _uav_att_param: fntsmc_param, _uav_pos_param: fntsmc_param):
        rl_base.__init__(self)
        uav_pos_ctrl.__init__(self, _uav_param, _uav_att_param, _uav_pos_param)
        
        self.staticGain = 2.0
        
        '''state limitation'''
        # 使用状态归一化，不用限制范围了
        '''state limitation'''
        
        '''rl_base'''
        self.name = 'uav_att_ctrl_RL'
        self.state_dim = 3 + 3  # phi theta psi p q r
        self.state_num = [np.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        
        self.current_state_norm = Normalization(self.state_dim)
        self.next_state_norm = Normalization(self.state_dim)
        
        self.current_state = np.zeros(self.state_dim)
        self.next_state = np.zeros(self.state_dim)
        
        self.action_dim = 3 + 3 + 3  # 3 for k1, 3 for k2, 3 for k4
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_range = [[0, 30.0] for _ in range(self.action_dim)]
        self.action_num = [np.inf for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.isActionContinuous = [True for _ in range(self.action_dim)]
        self.current_action = [0.0 for _ in range(self.action_dim)]
        
        self.reward = 0.
        self.sum_reward = 0.
        self.Q_pos = np.array([1., 1., 1.])  # 角度误差惩罚
        self.Q_vel = np.array([0.1, 0.1, 0.1])  # 角速度误差惩罚
        self.R = np.array([0.01, 0.01, 0.01])  # 期望加速度输出 (即控制输出) 惩罚
        self.is_terminal = False
        self.terminal_flag = 0
        '''rl_base'''
        
        '''opencv visualization for position control'''
        self.width = 1200
        self.height = 800
        self.x_offset = 40
        self.y_offset = 40
        self.offset = 20
        self.wp = (self.height - 2 * self.y_offset - 2 * self.offset) / 3
        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        dz = self.z_max - self.z_min
        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        self.pmt = (self.width - 2 * self.x_offset) / (self.time_max / self.dt)
        self.pmx = self.wp / dx
        self.pmy = self.wp / dy
        self.pmz = self.wp / dz
        self.pmx_p1 = self.wp / dx
        self.pmy_p1 = (self.height - 2 * self.y_offset) / dy
        self.pmx_p2 = self.wp / dy
        self.pmy_p2 = (self.height - 2 * self.y_offset) / dz
        self.pmx_p3 = self.wp / dz
        self.pmy_p3 = (self.height - 2 * self.y_offset) / dx
        '''opencv visualization for position control'''
    
    def dis2pixel(self, time, coord, flag: str, offset):
        t = self.x_offset + time / self.dt * self.pmt
        if flag == 'x':
            x = self.y_offset + self.wp - (coord - self.x_min) * self.pmx
            return int(t + offset[0]), int(x + offset[1])
        if flag == 'y':
            y = self.height - self.y_offset - self.wp - self.offset - (coord - self.y_min) * self.pmy
            return int(t + offset[0]), int(y + offset[1])
        if flag == 'z':
            z = self.height - self.y_offset - (coord - self.z_min) * self.pmz
            return int(t + offset[0]), int(z + offset[1])
        return offset
    
    def dis2pixel_trajectory_numpy2d(self, time, traj: np.ndarray, flag: str, offset: list) -> np.ndarray:
        """
        @param traj:        无人机轨迹，N * 3
        @param flag:        xyz
        @param offset:      偏移
        @return:
        @param time:
        """
        t = self.x_offset + time / self.dt * self.pmt
        if flag == 'x':
            x = self.y_offset + self.wp - (traj[:, 0] - self.x_min) * self.pmx + offset[0]
            return np.vstack((t, x)).T
        if flag == 'y':
            y = self.height - self.y_offset - self.offset - self.wp - (traj[:, 1] - self.y_min) * self.pmy + offset[0]
            return np.vstack((t, y)).T
        if flag == 'z':
            z = self.height - self.y_offset - (traj[:, 2] - self.z_min) * self.pmz + offset[1]
            return np.vstack((t, z)).T
        return np.array([])
    
    def draw_boundary_xot(self):
        cv.rectangle(self.image,
                     self.dis2pixel(0, self.x_min, 'x', [0, 0]),
                     self.dis2pixel(self.time_max, self.x_max, 'x', [0, 0]),
                     Color().Black, 2)
    
    def draw_boundary_yot(self):
        cv.rectangle(self.image,
                     self.dis2pixel(0, self.y_min, 'y', [0, 0]),
                     self.dis2pixel(self.time_max, self.y_max, 'y', [0, 0]),
                     Color().Black, 2)
    
    def draw_boundary_zot(self):
        cv.rectangle(self.image,
                     self.dis2pixel(0, self.z_min, 'z', [0, 0]),
                     self.dis2pixel(self.time_max, self.z_max, 'z', [0, 0]),
                     Color().Black, 2)
    
    def draw_boundary(self):
        self.draw_boundary_xot()
        self.draw_boundary_yot()
        self.draw_boundary_zot()
    
    def draw_label(self):
        pts = [self.dis2pixel(0, (self.x_max + self.x_min) / 2, 'x', [-35, 0]),
               self.dis2pixel(0, (self.y_max + self.y_min) / 2, 'y', [-35, 0]),
               self.dis2pixel(0, (self.z_max + self.z_min) / 2, 'z', [-35, 0]),
               self.dis2pixel(self.time_max / 2, self.z_min, 'z', [0, 20])]
        labels = ['X', 'Y', 'Z', 'Time']
        for _l, _pt in zip(labels, pts):
            cv.putText(self.image, _l, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
    
    def draw_region_grid(self, tNum: int, xNum: int, yNum: int, zNum: int):
        _dt = self.time_max / tNum
        _dx = (self.x_max - self.x_min) / xNum
        _dy = (self.y_max - self.y_min) / yNum
        _dz = (self.z_max - self.z_min) / zNum
        
        '''t'''
        for i in range(tNum - 1):
            ptx1 = self.dis2pixel((i + 1) * _dt, self.x_min, 'x', [0, 0])
            ptx2 = self.dis2pixel((i + 1) * _dt, self.x_max, 'x', [0, 0])
            cv.line(self.image, ptx1, ptx2, Color().Black, 1)
            pty1 = self.dis2pixel((i + 1) * _dt, self.y_min, 'y', [0, 0])
            pty2 = self.dis2pixel((i + 1) * _dt, self.y_max, 'y', [0, 0])
            cv.line(self.image, pty1, pty2, Color().Black, 1)
            ptz1 = self.dis2pixel((i + 1) * _dt, self.z_min, 'z', [0, 0])
            ptz2 = self.dis2pixel((i + 1) * _dt, self.z_max, 'z', [0, 0])
            cv.line(self.image, ptz1, ptz2, Color().Black, 1)
        
        '''X'''
        for i in range(xNum - 1):
            pt1 = self.dis2pixel(0, self.x_min + (i + 1) * _dx, 'x', [0, 0])
            pt2 = self.dis2pixel(self.time_max, self.x_min + (i + 1) * _dx, 'x', [0, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        
        '''Y'''
        for i in range(yNum - 1):
            pt1 = self.dis2pixel(0, self.y_min + (i + 1) * _dy, 'y', [0, 0])
            pt2 = self.dis2pixel(self.time_max, self.y_min + (i + 1) * _dy, 'y', [0, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        
        '''Z'''
        for i in range(zNum - 1):
            pt1 = self.dis2pixel(0, self.z_min + (i + 1) * _dz, 'z', [0, 0])
            pt2 = self.dis2pixel(self.time_max, self.z_min + (i + 1) * _dz, 'z', [0, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        
        self.draw_axis(tNum, xNum, yNum, zNum)
    
    def draw_axis(self, tNum: int, xNum: int, yNum: int, zNum: int):
        _dt = self.time_max / tNum
        _dx = (self.x_max - self.x_min) / xNum
        _dy = (self.y_max - self.y_min) / yNum
        _dz = (self.z_max - self.z_min) / zNum
        
        _t = np.linspace(0, self.time_max, tNum + 1)
        _x = np.linspace(self.x_min, self.x_max, xNum + 1)
        _y = np.linspace(self.y_min, self.y_max, yNum + 1)
        _z = np.linspace(self.z_min, self.z_max, zNum + 1)
        
        for __t in _t:
            if np.fabs(round(__t, 2) - int(__t)) < 0.01:
                _s = str(int(__t))
            else:
                _s = str(round(__t, 2))
            _pt = self.dis2pixel(__t, self.z_min, 'z', offset=[0, 15])
            
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
        
        for __x in _x:
            if np.fabs(round(__x, 2) - int(__x)) < 0.01:
                _s = str(int(__x))
            else:
                _s = str(round(__x, 2))
            _pt = self.dis2pixel(0, __x, 'x', offset=[-30 if len(_s) > 2 or __x < 0 else -15, 0])
            
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
        
        for __y in _y:
            if np.fabs(round(__y, 2) - int(__y)) < 0.01:
                _s = str(int(__y))
            else:
                _s = str(round(__y, 2))
            _pt = self.dis2pixel(0, __y, 'y', offset=[-30 if len(_s) > 2 or __y < 0 else -15, 0])
            
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
        
        for __z in _z:
            if np.fabs(round(__z, 2) - int(__z)) < 0.01:
                _s = str(int(__z))
            else:
                _s = str(round(__z, 2))
            _pt = self.dis2pixel(0, __z, 'z', offset=[-30 if len(_s) > 2 or __z < 0 else -15, 0])
            
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
    
    def draw_3d_points_projection(self, time, points: np.ndarray, colors: list):
        """
    @param colors:
    @param colors:
    @param points:
    @return:
    @param time:
    """
        '''X'''
        x = self.dis2pixel_trajectory_numpy2d(time, points, 'x', [0, 0])
        _l = x.shape[0]  # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(x[i][0])), int(round(x[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)
        
        '''Y'''
        y = self.dis2pixel_trajectory_numpy2d(time, points, 'y', [0, 0])
        _l = y.shape[0]  # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(y[i][0])), int(round(y[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)
        
        '''Z'''
        z = self.dis2pixel_trajectory_numpy2d(time, points, 'z', [0, 0])
        _l = z.shape[0]  # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(z[i][0])), int(round(z[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)
    
    def draw_3d_trajectory_projection(self, trajectory: np.ndarray):
        x = self.dis2pixel_trajectory_numpy2d(np.linspace(0, self.time_max, int(self.time_max / self.dt) + 1), trajectory, 'x', [0, 0])
        _l = x.shape[0]  # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(x[i][0])), int(round(x[i][1])))
            pt2 = (int(round(x[i + 1][0])), int(round(x[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)
        
        '''Y'''
        y = self.dis2pixel_trajectory_numpy2d(np.linspace(0, self.time_max, int(self.time_max / self.dt) + 1), trajectory, 'y', [0, 0])
        _l = y.shape[0]  # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(y[i][0])), int(round(y[i][1])))
            pt2 = (int(round(y[i + 1][0])), int(round(y[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)
        
        '''Z'''
        z = self.dis2pixel_trajectory_numpy2d(np.linspace(0, self.time_max, int(self.time_max / self.dt) + 1), trajectory, 'z', [0, 0])
        _l = z.shape[0]  # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(z[i][0])), int(round(z[i][1])))
            pt2 = (int(round(z[i + 1][0])), int(round(z[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)
    
    def draw_error(self, pos: np.ndarray, ref: np.ndarray):
        e = pos - ref
        v = self.dot_eta()
        att = rad2deg(self.rho1())
        _str = ('e = [%.2f, %.2f, %.2f]    v = [%.2f, %.2f, %.2f]    rho = [%.2f, %.2f, %.2f]' %
                (e[0], e[1], e[2], v[0], v[1], v[2], att[0], att[1], att[2]))
        cv.putText(self.image, _str, (self.x_offset, self.y_offset - 10), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple,
                   2)
        
        x = self.dis2pixel_trajectory_numpy2d(self.time_max/2, np.array([self.pos_max]), 'x', [0, 0])
        y = self.dis2pixel_trajectory_numpy2d(self.time_max/2, np.array([self.pos_max]), 'y', [0, 0])
        z = self.dis2pixel_trajectory_numpy2d(self.time_max/2, np.array([self.pos_max]), 'z', [0, 0])
        _str1 = 'ex: %.2f' % (e[0])
        _str2 = 'ey: %.2f' % (e[1])
        _str3 = 'ez: %.2f' % (e[2])
        cv.putText(self.image, _str1, (int(round(x[0][0])) - 30, int(round(x[0][1])) + 20), cv.FONT_HERSHEY_COMPLEX, 0.6,
				   Color().Chocolate2, 2)
        cv.putText(self.image, _str2, (int(round(y[0][0])) - 30, int(round(y[0][1])) + 20), cv.FONT_HERSHEY_COMPLEX, 0.6,
				   Color().Chocolate2, 2)
        cv.putText(self.image, _str3, (int(round(z[0][0])) - 30, int(round(z[0][1])) + 20), cv.FONT_HERSHEY_COMPLEX, 0.6,
				   Color().Chocolate2, 2)
    
    def draw_init_image(self):
        self.draw_3d_trajectory_projection(self.eta_d_all)
        self.draw_boundary()
        self.draw_label()
        self.draw_region_grid(5, 4, 4, 4)
        self.image_copy = self.image.copy()
    
    def show_pos_image(self, iswait: bool = False):
        if iswait:
            cv.imshow('Projection', self.image)
            cv.waitKey(0)
        else:
            cv.imshow('Projection', self.image)
            cv.waitKey(1)
    
    def visualization(self):
        self.image = self.image_copy.copy()
        self.draw_3d_points_projection(self.time, np.atleast_2d([self.eta_d]), [Color().DarkGreen])
        self.draw_3d_points_projection(self.time, np.atleast_2d([self.uav_pos()]), [Color().Red])
        self.draw_error(self.uav_pos(), self.eta_d)
        self.show_pos_image(False)
    
    def get_state(self) -> np.ndarray:
        e_pos_ = self.uav_pos() - self.eta_d
        e_vel_ = self.uav_vel() - self.dot_eta_d
        norm_state = np.concatenate((e_pos_, e_vel_))
        return norm_state
    
    def get_reward(self, param=None):
        """
        @param param:
        @return:
        """
        # ss = self.inverse_state_norm()
        # _e_pos = ss[0: 3]
        # _e_vel = ss[3: 6]
        _e_pos = self.uav_pos() - self.eta_d
        _e_vel = self.uav_vel() - self.dot_eta_d
        
        '''reward for position error'''
        u_pos = -np.dot(_e_pos ** 2, self.Q_pos)
        
        '''reward for velocity error'''
        u_vel = -np.dot(_e_vel ** 2, self.Q_vel)
        
        '''reward for control output'''
        u_acc = -np.dot(self.pos_ctrl.control_out ** 2, self.R)
        
        '''reward for att out!!'''
        u_extra = 0.
        if self.terminal_flag == 2 or self.terminal_flag == 3:  # 位置出界
            print('Position out')
            _n = (self.time_max - self.time) / self.dt - 1
            _u_x = _u_y = _u_z = 0.
            if self.x > self.x_max or self.x < self.x_min:
                _u_x = -(self.x_max - self.x_min) ** 2 * self.Q_pos[0]
            if self.y > self.y_max or self.y < self.y_min:
                _u_y = -(self.y_max - self.y_min) ** 2 * self.Q_pos[1]
            if self.z > self.z_max or self.z < self.z_min:
                _u_z = -(self.z_max - self.z_min) ** 2 * self.Q_pos[2]
            u_extra = _n * (_u_x + _u_y + _u_z + u_vel + u_acc)
        
        self.reward = u_pos + u_vel + u_acc + u_extra
        self.sum_reward += self.reward
    
    def is_success(self):
        '''
            跟踪控制，暂时不定义 “成功” 的概念，不好说啥叫成功，啥叫失败
            因此置为 False，实际也不调用这个函数即可，学习不成功可考虑再加
        '''
        return False
    
    def is_Terminal(self, param=None):
        self.terminal_flag = self.get_terminal_flag()
        if self.terminal_flag == 0:  # 普通状态
            self.is_terminal = False
        elif self.terminal_flag == 1:  # 超时
            self.is_terminal = True
        elif self.terminal_flag == 2:  # 位置
            self.is_terminal = True
        elif self.terminal_flag == 3:  # 姿态
            self.is_terminal = True
        else:
            self.is_terminal = False
    
    def get_terminal_flag(self) -> int:
        self.terminal_flag = 0
        if self.is_pos_out():
            # print('Position out...')
            self.terminal_flag = 2
        if self.is_att_out():
            # print('Attitude out...')
            self.terminal_flag = 3
        if self.time > self.time_max - self.dt / 2:
            # print('Time out...')
            self.terminal_flag = 1
        return self.terminal_flag
    
    def step_update(self, action: list):
        """
        @param action:	这个 action 其实是油门 + 三个力矩
        @return:
        """
        self.current_action = np.array(action)
        self.current_state = self.get_state()
        
        self.update(action=self.current_action)
        self.is_Terminal()
        self.next_state = self.get_state()
        self.get_reward()
    
    def get_param_from_actor(self, action_from_actor: np.ndarray, hehe_flag: bool = True):
        if np.min(action_from_actor) < 0:
            print('ERROR!!!!')
        if hehe_flag:
            for i in range(3):  # 分别对应 k1: 0 1 2, k2: 3 4 5, k4: 6 7 8
                if action_from_actor[i] > 0:
                    self.pos_ctrl.k1[i] = action_from_actor[i]
                if action_from_actor[i + 3] > 0:
                    self.pos_ctrl.k2[i] = action_from_actor[i + 3]
                if action_from_actor[i + 6] > 0:
                    self.pos_ctrl.k4[i] = action_from_actor[i + 6] * 5
        else:
            for i in range(3):  # 分别对应 k1: 0 1 2, k2: 3 4 5, k4: 6 7 8
                if action_from_actor[i] > 0:
                    self.pos_ctrl.k1[i] = action_from_actor[i]
                if action_from_actor[i + 3] > 0:
                    self.pos_ctrl.k2[i] = action_from_actor[i + 3]
                if action_from_actor[i + 6] > 0:
                    self.pos_ctrl.k4[i] = action_from_actor[i + 6]
    
    def save_state_norm(self, path, msg=None):
        data = {
            'cur_n': self.current_state_norm.running_ms.n * np.ones(self.state_dim),
            'cur_mean': self.current_state_norm.running_ms.mean,
            'cur_std': self.current_state_norm.running_ms.std,
            'cur_S': self.current_state_norm.running_ms.S,
            'next_n': self.next_state_norm.running_ms.n * np.ones(self.state_dim),
            'next_mean': self.next_state_norm.running_ms.mean,
            'next_std': self.next_state_norm.running_ms.std,
            'next_S': self.next_state_norm.running_ms.S,
        }
        if msg is None:
            pd.DataFrame(data).to_csv(path + 'state_norm.csv', index=False)
        else:
            pd.DataFrame(data).to_csv(path + 'state_norm_' + msg + '.csv', index=False)
    
    def state_norm_batch(self, cur_data: np.ndarray, next_data: np.ndarray):
        ll = len(cur_data)
        for i in range(ll):
            cur_data[i] = self.current_state_norm(cur_data[i], update=True)
            next_data[i] = self.next_state_norm(next_data[i], update=True)
        return cur_data, next_data
    
    def load_norm_normalizer_from_file(self, path, file):
        data = pd.read_csv(path + file, header=0).to_numpy()
        self.current_state_norm.running_ms.n = data[0, 0]
        self.current_state_norm.running_ms.mean = data[:, 1]
        self.current_state_norm.running_ms.std = data[:, 2]
        self.current_state_norm.running_ms.S = data[:, 3]
        self.next_state_norm.running_ms.n = data[0, 4]
        self.next_state_norm.running_ms.mean = data[:, 5]
        self.next_state_norm.running_ms.std = data[:, 6]
        self.next_state_norm.running_ms.S = data[:, 7]
    
    def reset_env(self,
                  random_pos_trajectory: bool = False,
                  random_pos0: bool = False,
                  yaw_fixed: bool = False,
                  new_pos_ctrl_param: fntsmc_param = None,
                  outer_param: list = None,
                  is_ideal:bool=False
                  ):
        self.reset_uav()
        self.collector_reset()
        self.generate_random_pos_trajectory(is_random=random_pos_trajectory,
                                            random_pos0=random_pos0,
                                            yaw_fixed=yaw_fixed,
                                            outer_param=outer_param)
        self.generate_uncertainty(is_ideal=is_ideal)
        self.reward = 0.
        self.sum_reward = 0.
        self.is_terminal = False
        self.terminal_flag = 0
        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        
        self.att_ctrl.fntsmc_reset()
        
        if new_pos_ctrl_param is not None:
            self.pos_ctrl.fntsmc_reset_with_new_param(new_pos_ctrl_param)
        else:
            self.pos_ctrl.fntsmc_reset()
        self.draw_3d_trajectory_projection(self.eta_d_all)
        self.draw_init_image()
