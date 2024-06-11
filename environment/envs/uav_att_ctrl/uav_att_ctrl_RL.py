import sys, os

from environment.envs.UAV.uav import uav_param
from environment.envs.UAV.uav_att_ctrl import uav_att_ctrl, fntsmc_param
from environment.color import Color
from environment.envs.UAV.ref_cmd import *

from algorithm.rl_base.rl_base import rl_base

from utils.classes import Normalization
import cv2 as cv
import pandas as pd


class uav_att_ctrl_RL(rl_base, uav_att_ctrl):
	def __init__(self, _uav_param: uav_param, _uav_att_param: fntsmc_param):
		rl_base.__init__(self)
		uav_att_ctrl.__init__(self, _uav_param, _uav_att_param)
		
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
		self.Q_att = np.array([1., 1., 1.])        # 角度误差惩罚
		self.Q_pqr = np.array([0.01, 0.01, 0.01])  # 角速度误差惩罚
		self.R = np.array([0.01, 0.01, 0.01])      # 期望加速度输出 (即控制输出) 惩罚
		self.is_terminal = False
		self.terminal_flag = 0
		'''rl_base'''
		
		'''opencv visualization for attitude control'''
		self.att_w = 900
		self.att_h = 300
		self.att_offset = 10  # 图与图之间的间隔
		self.att_image = np.ones([self.att_h, self.att_w, 3], np.uint8) * 255
		self.att_image_copy = self.att_image.copy()
		self.att_image_r = int(0.35 * self.att_w / 3)
		'''opencv visualization for attitude control'''
	
	def draw_init_image(self):
		x1 = int(self.att_w / 3)
		x2 = int(2 * self.att_w / 3)
		y = int(self.att_h / 2) + 15

		c = [(int(x1 / 2), y), (x1 + int(x1 / 2), y), (2 * x1 + int(x1 / 2), y)]

		cv.line(self.att_image, (x1, 0), (x1, self.att_h), Color().Black, 1, cv.LINE_AA)
		cv.line(self.att_image, (x2, 0), (x2, self.att_h), Color().Black, 1, cv.LINE_AA)

		for _c in c:
			cv.circle(self.att_image, _c, self.att_image_r, Color().Orange, 2, cv.LINE_AA)
			cv.circle(self.att_image, _c, 5, Color().Black, -1)
			cv.line(self.att_image, (_c[0], _c[1] + self.att_image_r), (_c[0], _c[1] - self.att_image_r), Color().Black, 1, cv.LINE_AA)
			cv.line(self.att_image, (_c[0] - self.att_image_r, _c[1]), (_c[0] + self.att_image_r, _c[1]), Color().Black, 1, cv.LINE_AA)
			cv.putText(self.att_image, '0', (_c[0] - 7, _c[1] - self.att_image_r - 5), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Red, 1)
			cv.putText(self.att_image, '-90', (_c[0] - self.att_image_r - 45, _c[1] + 4), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Red, 1)
			cv.putText(self.att_image, '90', (_c[0] + self.att_image_r + 7, _c[1] + 4), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Red, 1)
			cv.putText(self.att_image, '-180', (_c[0] - 60, _c[1] + self.att_image_r + 15), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Red, 1)
			cv.putText(self.att_image, '180', (_c[0] + 5, _c[1] + self.att_image_r + 15), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Red, 1)

		cv.putText(self.att_image, 'roll', (int(x1 / 2 - 20), 25), cv.FONT_HERSHEY_COMPLEX, 0.8, Color().Blue, 1)
		cv.putText(self.att_image, 'pitch', (int(x1 + x1 / 2 - 28), 25), cv.FONT_HERSHEY_COMPLEX, 0.8, Color().Blue, 1)
		cv.putText(self.att_image, 'yaw', (int(2 * x1 + x1 / 2 - 20), 25), cv.FONT_HERSHEY_COMPLEX, 0.8, Color().Blue, 1)

		self.att_image_copy = self.att_image.copy()

	def draw_att(self):
		x1 = int(self.att_w / 3)
		y = int(self.att_h / 2) + 15
		c = [(int(x1 / 2), y), (x1 + int(x1 / 2), y), (2 * x1 + int(x1 / 2), y)]

		for _c, _a, _ref_a in zip(c, self.uav_att(), self.rho_d_all[self.n]):
			px = _c[0] + int(self.att_image_r * np.cos(np.pi / 2 - _a))
			py = _c[1] - int(self.att_image_r * np.sin(np.pi / 2 - _a))
			px2 = _c[0] + int(self.att_image_r * np.cos(np.pi / 2 - _ref_a))
			py2 = _c[1] - int(self.att_image_r * np.sin(np.pi / 2 - _ref_a))
			_e = (_ref_a - _a) * 180 / np.pi
			_r = (_c[0] + self.att_image_r - 55, _c[1] - self.att_image_r - 15)
			cv.line(self.att_image, _c, (px, py), Color().Blue, 2, cv.LINE_AA)
			cv.line(self.att_image, _c, (px2, py2), Color().Red, 2, cv.LINE_AA)
			cv.putText(self.att_image, 'e: %.1f' % _e, _r, cv.FONT_HERSHEY_COMPLEX, 0.7, Color().Purple, 1)
			cv.putText(self.att_image, '%.1f' % (_a * 180 / np.pi), (px, py), cv.FONT_HERSHEY_COMPLEX, 0.7, Color().Purple, 1)

		_str = 't = %.2f' % self.time
		_r2 = (c[0][0] - self.att_image_r - 40, c[0][1] - self.att_image_r - 15)
		cv.putText(self.att_image, _str, _r2, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 1)

	def show_att_image(self, iswait: bool = False):
		if iswait:
			cv.imshow('Attitude', self.att_image)
			cv.waitKey(0)
		else:
			cv.imshow('Attitude', self.att_image)
			cv.waitKey(1)
	
	def visualization(self):
		self.att_image = self.att_image_copy.copy()
		self.draw_att()
		self.show_att_image(iswait=False)
	
	def get_state(self) -> np.ndarray:
		e_att_ = self.uav_att() - self.ref
		e_pqr_ = self.uav_dot_att() - self.dot_ref
		state = np.concatenate((e_att_, e_pqr_))
		return state
	
	def get_reward(self, param=None):
		"""
		@param param:
		@return:
		"""
		_e_att = self.uav_att() - self.ref
		_e_pqr = self.uav_dot_att() - self.dot_ref
		
		'''reward for position error'''
		u_att = -np.dot(_e_att ** 2, self.Q_att)
		
		'''reward for velocity error'''
		u_pqr = -np.dot(_e_pqr ** 2, self.Q_pqr)
		
		'''reward for control output'''
		u_acc = -np.dot(self.att_ctrl.control_in ** 2, self.R)
		
		'''reward for att out!!'''
		u_extra = 0.
		if self.terminal_flag == 3:
			print('Attitude out')
			'''
				给出界时刻的位置、速度、输出误差的累计
			'''
			_n = (self.time_max - self.time) / self.dt - 1
			'''这里把三个姿态的累计惩罚分开'''
			_u_phi = _u_theta = _u_psi = 0.
			if self.phi > self.phi_max or self.phi < self.phi_min:
				# _u_phi = -(_e_att[0] ** 2 * self.Q_att[0] + _e_pqr[0] ** 2 * self.Q_pqr[0])
				_u_phi = -np.pi ** 2 * self.Q_att[0]
			if self.theta > self.theta_max or self.theta < self.theta_min:
				# _u_theta = -(_e_att[1] ** 2 * self.Q_att[1] + _e_pqr[1] ** 2 * self.Q_pqr[1])
				_u_theta = -np.pi ** 2 * self.Q_att[1]
			if self.psi > self.psi_max or self.psi < self.psi_min:
				# _u_psi = -(_e_att[2] ** 2 * self.Q_att[2] + _e_pqr[2] ** 2 * self.Q_pqr[2])
				_u_psi = -4 * np.pi ** 2 * self.Q_att[2]
			
			u_extra = _n * (_u_phi + _u_theta + _u_psi + u_pqr + u_acc + u_att)
		
		self.reward = u_att + u_pqr + u_acc + u_extra
		self.sum_reward += self.reward
	
	def is_success(self):
		"""
		@return:
		"""
		'''
			跟踪控制，暂时不定义 “成功” 的概念，不好说啥叫成功，啥叫失败
			因此置为 False，实际也不调用这个函数即可，学习不成功可考虑再加
		'''
		return False

	def get_terminal_flag(self) -> int:
		self.terminal_flag = 0
		if self.is_att_out():
			# print('Attitude out...')
			self.terminal_flag = 3
		if self.time > self.time_max - self.dt / 2:
			# print('Time out...')
			self.terminal_flag = 1
		return self.terminal_flag
	
	def is_Terminal(self, param=None):
		self.terminal_flag = self.get_terminal_flag()
		if self.terminal_flag == 0 or self.terminal_flag == 2:  # 普通状态
			self.is_terminal = False
		elif self.terminal_flag == 1:  # 超时
			self.is_terminal = True
		elif self.terminal_flag == 3:  # 姿态
			self.is_terminal = True
		else:
			self.is_terminal = False
	
	def step_update(self, action: list):
		"""
		@param action:	这个 action 是三个力矩
		@return:
		"""
		self.current_action = np.array(action)
		self.current_state = self.get_state()
		self.update(action=self.current_action)
		self.is_Terminal()
		self.next_state = self.get_state()
		self.get_reward()
	
	def get_param_from_actor(self, action_from_actor: np.ndarray, hehe_flag: bool = True):
		"""
		@param action_from_actor:
		@return:
		"""
		if np.min(action_from_actor) < 0:
			print('ERROR!!!!')
		if hehe_flag:
			for i in range(3):  # 分别对应 k1: 0 1 2, k2: 3 4 5, k4: 6 7 8
				if action_from_actor[i] > 0:
					self.att_ctrl.k1[i] = action_from_actor[i] * 5
				if action_from_actor[i + 3] > 0:
					self.att_ctrl.k2[i] = action_from_actor[i + 3]
				if action_from_actor[i + 6] > 0:
					self.att_ctrl.k4[i] = action_from_actor[i + 6] * 5
		else:
			for i in range(3):	# 分别对应 k1: 0 1 2, k2: 3 4 5, k4: 6 7 8
				if action_from_actor[i] > 0:
					self.att_ctrl.k1[i] = action_from_actor[i]
				if action_from_actor[i + 3] > 0:
					self.att_ctrl.k2[i] = action_from_actor[i + 3]
				if action_from_actor[i + 6] > 0:
					self.att_ctrl.k4[i] = action_from_actor[i + 6]
		
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
				  random_att_trajectory: bool = False,			# 是否随机生成姿态参考指令
				  yaw_fixed: bool = False,						# 是否固定偏航角
				  new_att_ctrl_param: fntsmc_param = None,		# 是否有新的控制器参数
				  outer_param: list = None						# 是否有外部参数输入
				  ):
		self.reset_uav()
		self.collector_reset()
		self.generate_random_att_trajectory(is_random=random_att_trajectory, yaw_fixed=yaw_fixed, outer_param=outer_param)
		
		self.reward = 0.
		self.sum_reward = 0.
		self.is_terminal = False
		self.terminal_flag = 0
		self.att_image = np.ones([self.att_h, self.att_w, 3], np.uint8) * 255
		self.att_image_copy = self.att_image.copy()
		
		if new_att_ctrl_param is not None:
			self.att_ctrl.fntsmc_reset_with_new_param(new_att_ctrl_param)
		else:
			self.att_ctrl.fntsmc_reset()
		self.draw_init_image()