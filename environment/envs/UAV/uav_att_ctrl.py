import numpy as np

from environment.envs.UAV.collector import data_collector
from environment.envs.UAV.FNTSMC import fntsmc, fntsmc_param
from environment.envs.UAV.uav import UAV, uav_param
from environment.envs.UAV.ref_cmd import *
from utils.functions import *


class uav_att_ctrl(UAV):
	def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param):
		super(uav_att_ctrl, self).__init__(UAV_param)
		self.att_ctrl = fntsmc(att_ctrl_param)
		self.collector = data_collector(int(np.round(self.time_max / self.dt)))
		self.ref = np.zeros(3)
		self.dot_ref = np.zeros(3)
		self.dot2_ref = np.zeros(3)
		
		'''参考轨迹记录'''
		self.ref_att_amplitude = np.zeros(3)
		self.ref_att_period = np.zeros(3)
		self.ref_att_bias_a = np.zeros(3)
		self.ref_att_bias_phase = np.zeros(3)
		self.rho_d_all = np.atleast_2d([])
		self.dot_rho_d_all = np.atleast_2d([])
		self.dot2_rho_d_all = np.atleast_2d([])
		'''参考轨迹记录'''
		
		self.uncertainty = np.atleast_2d([])
		
	def att_control(self, att_only: bool = True, obs: np.ndarray = np.zeros(3)):
		self.ref = self.rho_d_all[self.n]
		self.dot_ref = self.dot_rho_d_all[self.n]
		self.dot2_ref = self.dot2_rho_d_all[self.n]
		e_rho = self.rho1() - self.ref
		dot_e_rho = self.dot_rho1() - self.dot_ref
		self.att_ctrl.control_update_inner(e_rho=e_rho,
										   dot_e_rho=dot_e_rho,
										   dd_ref=self.dot2_ref,
										   W=self.W(),
										   dW=self.dW(),
										   omega=self.omega(),
										   A_omega=self.A_omega(),
										   B_omega=self.B_omega(),
										   obs=obs,
										   att_only=att_only)
		return self.att_ctrl.control_in
	
	def update(self, action:np.ndarray):
		action_4_uav = np.insert(action, 0, self.m * self.g / (np.cos(self.phi) * np.cos(self.theta)))
		data_block = {'time': self.time,
					  'control': action_4_uav,
					  'ref_angle': self.ref,
					  'ref_pos': np.array([0., 0., 0.]),
					  'ref_vel': np.array([0., 0., 0.]),
					  'd_in': np.zeros(3),
					  'd_in_obs': np.zeros(3),
					  'd_in_e_1st': np.zeros(3),
					  'd_out': np.zeros(3),
					  'd_out_obs': np.zeros(3),
					  'd_out_e_1st': np.zeros(3),
					  'state': np.hstack((np.zeros(6), self.uav_att_pqr_call_back()))}
		self.collector.record(data_block)
		dis = np.concatenate((np.zeros(3), self.uncertainty[self.n]))
		self.rk44(action=action_4_uav, dis=dis, n=1, att_only=True)
	
	def generate_ref_att_trajectory(self, _amplitude: np.ndarray, _period: np.ndarray, _bias_a: np.ndarray, _bias_phase: np.ndarray):
		"""
		@param _amplitude:
		@param _period:
		@param _bias_a:
		@param _bias_phase:
		@return:
		"""
		t = np.linspace(0, self.time_max, int(self.time_max / self.dt) + 1)
		w = 2 * np.pi / _period
		r_phi = _bias_a[0] + _amplitude[0] * np.sin(w[0] * t + _bias_phase[0])
		r_theta = _bias_a[1] + _amplitude[1] * np.sin(w[1] * t + _bias_phase[1])
		r_psi = _bias_a[2] + _amplitude[2] * np.sin(w[2] * t + _bias_phase[2])
		
		r_d_phi = _amplitude[0] * w[0] * np.cos(w[0] * t + _bias_phase[0])
		r_d_theta = _amplitude[1] * w[1] * np.cos(w[1] * t + _bias_phase[1])
		r_d_psi = _amplitude[2] * w[2] * np.cos(w[2] * t + _bias_phase[2])
		
		r_dd_phi = -_amplitude[0] * w[0] ** 2 * np.sin(w[0] * t + _bias_phase[0])
		r_dd_theta = -_amplitude[1] * w[1] ** 2 * np.sin(w[1] * t + _bias_phase[1])
		r_dd_psi = -_amplitude[2] * w[2] ** 2 * np.sin(w[2] * t + _bias_phase[2])
		
		return np.vstack((r_phi, r_theta, r_psi)).T, np.vstack((r_d_phi, r_d_theta, r_d_psi)).T, np.vstack((r_dd_phi, r_dd_theta, r_dd_psi)).T
	
	def generate_random_att_trajectory(self, is_random: bool = False, yaw_fixed: bool = False, outer_param: list = None):
		"""
		@param is_random:       random trajectory or not
		@param yaw_fixed:       fix the yaw angle or not
		@param outer_param:     choose whether accept user-defined trajectory parameters or not
		@return:                None
		"""
		if outer_param is not None:
			A = outer_param[0]
			T = outer_param[1]
			phi0 = outer_param[2]
		else:
			if is_random:
				if np.random.uniform(0, 1) < 0.7:		# 70% 的概率选择之前不好的区域
					A = np.random.uniform(low=deg2rad(30), high=deg2rad(65)) * np.ones(3)
					T = np.random.uniform(low=3, high=5) * np.ones(3)
				else:
					A = np.random.uniform(low=deg2rad(5), high=deg2rad(30)) * np.ones(3)
					T = np.random.uniform(low=5, high=7) * np.ones(3)
				# A = np.random.uniform(low=deg2rad(5), high=deg2rad(65)) * np.ones(3)
				# # A = np.array([
				# # 	np.random.uniform(low=0, high=self.phi_max if self.phi_max < np.pi / 3 else np.pi / 3),
				# # 	np.random.uniform(low=0, high=self.theta_max if self.theta_max < np.pi / 3 else np.pi / 3),
				# # 	np.random.uniform(low=0, high=self.psi_max if self.psi_max < np.pi / 2 else np.pi / 2)])
				#
				# # T = np.random.uniform(low=3, high=6, size=3)  # 随机生成周期
				# T = np.random.uniform(low=2, high=7) * np.ones(3)
				# # phi0 = np.random.uniform(low=0, high=np.pi / 2, size=3)
				phi0 = np.array([0, 0, 0])
			else:
				A = np.array([np.pi / 3, np.pi / 3, np.pi / 3])
				T = np.array([3, 3, 3])
				phi0 = np.array([0., 0., 0.])
			if yaw_fixed:
				A[2] = 0.
				phi0[2] = 0.
		
		self.ref_att_amplitude = A
		self.ref_att_period = T
		self.ref_att_bias_a = np.zeros(3)
		self.ref_att_bias_phase = phi0
		self.rho_d_all, self.dot_rho_d_all, self.dot2_rho_d_all = (
			self.generate_ref_att_trajectory(self.ref_att_amplitude, self.ref_att_period, self.ref_att_bias_a, self.ref_att_bias_phase))
	
	def generate_uncertainty(self, is_ideal: bool = False):
		N = int(self.time_max / self.dt) + 1
		t = np.linspace(0, self.time_max, N)
		if is_ideal:
			self.uncertainty = np.zeros((N, 3))
		else:
			self.uncertainty = np.zeros((N, 3))
			T = 3
			w = 2 * np.pi / T
			self.uncertainty[:, 0] = 1.5 * np.sin(w * t) + 0.2 * np.cos(3 * w * t)
			self.uncertainty[:, 1] = 0.5 * np.cos(2 * w * t) + 0.15 * np.sin(w * t)
			self.uncertainty[:, 2] = 0.8 * np.sin(2 * w * t) + 1.0 * np.cos(2 * w * t)
	
	def controller_reset(self):
		self.ref_att_amplitude = np.zeros(3)
		self.ref_att_period = np.zeros(3)
		self.ref_att_bias_a = np.zeros(3)
		self.ref_att_bias_phase = np.zeros(3)
		self.rho_d_all = np.atleast_2d([])
		self.dot_rho_d_all = np.atleast_2d([])
		self.dot2_rho_d_all = np.atleast_2d([])
		self.att_ctrl.fntsmc_reset()
	
	def controller_reset_with_new_param(self, new_att_param: fntsmc_param = None):
		'''参考轨迹记录'''
		self.ref_att_amplitude = None
		self.ref_att_period = None
		self.ref_att_bias_a = None
		self.ref_att_bias_phase = None
		self.rho_d_all = None
		self.dot_rho_d_all = None
		self.dot2_rho_d_all = None
		'''参考轨迹记录'''
		if new_att_param is not None:
			self.att_ctrl.fntsmc_reset_with_new_param(new_att_param)
	
	def collector_reset(self):
		self.collector.reset()
