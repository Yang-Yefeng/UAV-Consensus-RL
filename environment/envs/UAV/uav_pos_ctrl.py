import numpy as np

from environment.envs.UAV.collector import data_collector
from environment.envs.UAV.FNTSMC import fntsmc, fntsmc_param
from environment.envs.UAV.uav import UAV, uav_param
from environment.envs.UAV.ref_cmd import *
from utils.functions import *


class uav_pos_ctrl(UAV):
	def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param, pos_ctrl_param: fntsmc_param):
		super(uav_pos_ctrl, self).__init__(UAV_param)
		self.att_ctrl = fntsmc(att_ctrl_param)
		self.pos_ctrl = fntsmc(pos_ctrl_param)
		
		self.collector = data_collector(int(np.round(self.time_max / self.dt)))
		self.eta_d = np.zeros(3)
		self.dot_eta_d = np.zeros(3)
		self.dot2_eta_d = np.zeros(3)
		self.rho_d = np.zeros(3)
		self.dot_rho_d = np.zeros(3)
		
		'''参考轨迹记录'''
		self.ref_att_amplitude = np.zeros(3)
		self.ref_att_period = np.zeros(3)
		self.ref_att_bias_a = np.zeros(3)
		self.ref_att_bias_phase = np.zeros(3)
		self.eta_d_all = np.atleast_2d([])
		self.dot_eta_d_all = np.atleast_2d([])
		self.dot2_eta_d_all = np.atleast_2d([])
		'''参考轨迹记录'''
	
	def att_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray, att_only: bool = True):
		self.ref = ref
		self.dot_ref = dot_ref
		self.dot2_ref = dot2_ref
		e_rho = self.rho1() - self.ref
		dot_e_rho = self.dot_rho1() - self.dot_ref
		self.att_ctrl.control_update_inner(e_rho=e_rho,
										   dot_e_rho=dot_e_rho,
										   dd_ref=dot2_ref,
										   W=self.W(),
										   dW=self.dW(),
										   omega=self.omega(),
										   A_omega=self.A_omega(),
										   B_omega=self.B_omega(),
										   obs=np.zeros(3),
										   att_only=att_only)
		return self.att_ctrl.control_in
		
	def pos_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray):
		self.eta_d = ref
		self.dot_eta_d = dot_ref
		self.dot2_eta_d = dot2_ref
		e_eta = self.eta() - self.eta_d