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
        self.ref_pos_amplitude = np.zeros(3)
        self.ref_pos_period = np.zeros(3)
        self.ref_pos_bias_a = np.zeros(3)
        self.ref_pos_bias_phase = np.zeros(3)
        self.eta_d_all = np.atleast_2d([])
        self.dot_eta_d_all = np.atleast_2d([])
        self.dot2_eta_d_all = np.atleast_2d([])
        self.psi_d_all = np.array([])
        self.dot_psi_d_all = np.array([])
        self.dot2_psi_d_all = np.array([])
        '''参考轨迹记录'''
    
    def att_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray, att_only: bool = True):
        e_rho = self.rho1() - ref
        dot_e_rho = self.dot_rho1() - dot_ref
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
    
    def pos_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray, obs: np.ndarray, att_lim=None):
        self.eta_d = ref
        self.dot_eta_d = dot_ref
        self.dot2_eta_d = dot2_ref
        e_eta = self.eta() - self.eta_d
        dot_e_eta = self.dot_eta() - self.dot_eta_d
        self.pos_ctrl.control_update_outer(e_eta, dot_e_eta, self.dot_eta(), self.kt, self.m, self.dot2_eta_d, obs)
        phi_d, theta_d, uf = self.uo_2_ref_angle_throttle(att_lim=att_lim)
        return phi_d, theta_d, uf
    
    def uo_2_ref_angle_throttle(self, att_lim=None):
        ux = self.pos_ctrl.control_out[0]
        uy = self.pos_ctrl.control_out[1]
        uz = self.pos_ctrl.control_out[2]
        uf = (uz + self.g) * self.m / (np.cos(self.phi) * np.cos(self.theta))
        
        asin_phi_d = min(max((ux * np.sin(self.psi) - uy * np.cos(self.psi)) * self.m / uf, -1), 1)
        phi_d = np.arcsin(asin_phi_d)
        if att_lim is not None:
            phi_d = np.clip(phi_d, -att_lim[0], att_lim[0])
        
        asin_theta_d = min(max((ux * np.cos(self.psi) + uy * np.sin(self.psi)) * self.m / (uf * np.cos(phi_d)), -1), 1)
        theta_d = np.arcsin(asin_theta_d)
        if att_lim is not None:
            theta_d = np.clip(theta_d, -att_lim[1], att_lim[1])
        
        return phi_d, theta_d, uf  # TODO
    
    def generate_action_4_uav(self, att_lim=None, dot_att_lim=None):
        eta_d = self.eta_d_all[self.n]
        dot_eta_d = self.dot_eta_d_all[self.n]
        dot2_eta_d = self.dot2_eta_d_all[self.n]
        
        phi_d, theta_d, uf = self.pos_control(eta_d, dot_eta_d, dot2_eta_d, obs=np.zeros(3), att_lim=att_lim)
        dot_phi_d = (phi_d - self.rho_d[0]) / self.dt
        if dot_att_lim is not None:
            if dot_phi_d > dot_att_lim[0]:
                dot_phi_d = dot_att_lim[0]
            # phi_d = dot_phi_d * self.dt + self.rho_d[0]
            if dot_phi_d < -dot_att_lim[0]:
                dot_phi_d = -dot_att_lim[0]
            # phi_d = dot_phi_d * self.dt + self.rho_d[0]
        
        dot_theta_d = (theta_d - self.rho_d[1]) / self.dt
        if dot_att_lim is not None:
            if dot_theta_d > dot_att_lim[1]:
                dot_theta_d = dot_att_lim[1]
            # theta_d = dot_theta_d * self.dt + self.rho_d[1]
            if dot_theta_d < -dot_att_lim[1]:
                dot_theta_d = -dot_att_lim[1]
            # theta_d = dot_theta_d * self.dt + self.rho_d[1]
        
        self.rho_d = np.array([phi_d, theta_d, self.psi_d_all[self.n]])
        self.dot_rho_d = np.array([dot_phi_d, dot_theta_d, self.dot_psi_d_all[self.n]])
        
        torque = self.att_control(self.rho_d, self.dot_rho_d, np.zeros(3), False)
        
        action_4_uav = [uf, torque[0], torque[1], torque[2]]
        return action_4_uav
    
    def update(self, action: np.ndarray):
        data_block = {'time': self.time,
                      'control': action,
                      'ref_angle': self.rho_d,
                      'ref_pos': self.eta_d,
                      'ref_vel': self.dot_eta_d,
                      'd_in': np.zeros(3),
                      'd_in_obs': np.zeros(3),
                      'd_in_e_1st': np.zeros(3),
                      'd_out': np.zeros(3),
                      'd_out_obs': np.zeros(3),
                      'd_out_e_1st': np.zeros(3),
                      'state': np.hstack((np.zeros(6), self.uav_att_pqr_call_back()))}
        self.collector.record(data_block)
        self.rk44(action=action, dis=np.zeros(6), n=1, att_only=False)
    
    def generate_ref_pos_trajectory(self, _amplitude: np.ndarray, _period: np.ndarray, _bias_a: np.ndarray, _bias_phase: np.ndarray):
        """
		@param _amplitude:
		@param _period:
		@param _bias_a:
		@param _bias_phase:
		@return:
		"""
        t = np.linspace(0, self.time_max, int(self.time_max / self.dt) + 1)
        w = 2 * np.pi / _period
        r_x = _bias_a[0] + _amplitude[0] * np.sin(w[0] * t + _bias_phase[0])
        r_y = _bias_a[1] + _amplitude[1] * np.sin(w[1] * t + _bias_phase[1])
        r_z = _bias_a[2] + _amplitude[2] * np.sin(w[2] * t + _bias_phase[2])
        self.psi_d_all = _bias_a[3] + _amplitude[3] * np.sin(w[3] * t + _bias_phase[3])
        
        r_d_x = _amplitude[0] * w[0] * np.cos(w[0] * t + _bias_phase[0])
        r_d_y = _amplitude[1] * w[1] * np.cos(w[1] * t + _bias_phase[1])
        r_d_z = _amplitude[2] * w[2] * np.cos(w[2] * t + _bias_phase[2])
        self.dot_psi_d_all = _amplitude[3] * w[3] * np.cos(w[3] * t + _bias_phase[3])
        
        r_dd_x = -_amplitude[0] * w[0] ** 2 * np.sin(w[0] * t + _bias_phase[0])
        r_dd_y = -_amplitude[1] * w[1] ** 2 * np.sin(w[1] * t + _bias_phase[1])
        r_dd_z = -_amplitude[2] * w[2] ** 2 * np.sin(w[2] * t + _bias_phase[2])
        self.dot2_psi_d_all = -_amplitude[3] * w[3] ** 2 * np.sin(w[3] * t + _bias_phase[3])
        
        self.eta_d_all = np.vstack((r_x, r_y, r_z)).T
        self.dot_eta_d_all = np.vstack((r_d_x, r_d_y, r_d_z)).T
        self.dot2_eta_d_all = np.vstack((r_dd_x, r_dd_y, r_dd_z)).T
    
    def generate_random_pos_trajectory(self, is_random: bool, random_pos0: bool, yaw_fixed: bool, outer_param=None):
        if outer_param is not None:
            A = np.array(outer_param[0])
            T = np.array(outer_param[1])
            phi0 = np.array(outer_param[2])
            ba = np.array(outer_param[3])
        else:
            if is_random:
                A = np.array([
                    np.random.uniform(low=0, high=3),  # x
                    np.random.uniform(low=0, high=3),  # y
                    np.random.uniform(low=0, high=3),  # z
                    np.random.uniform(low=0, high=deg2rad(80))])  # psi
                T = np.random.uniform(low=4, high=8, size=4)
                phi0 = np.random.uniform(low=0, high=np.pi / 2, size=4)
                ba = np.zeros(4)
            
            # A = np.concatenate((np.random.uniform(low=0, high=3) * np.ones(3),
            # 					[np.random.uniform(low=0, high=deg2rad(70))]))
            # T = np.random.uniform(low=4, high=8) * np.ones(4)
            # phi0 = np.random.uniform(low=0, high=np.pi / 2) * np.ones(4)
            # ba = np.zeros(4)
            else:
                A = np.array([2, 2, 2, deg2rad(80)])
                T = np.array([5, 5, 5, 5])
                phi0 = np.array([np.pi / 2, 0., 0., 0.])
                ba = np.zeros(4)
            if yaw_fixed:
                A[3] = 0.
                phi0[3] = 0.
        self.ref_pos_amplitude = A
        self.ref_pos_period = T
        self.ref_pos_bias_a = ba
        self.ref_pos_bias_phase = phi0
        if random_pos0:
            self.x = np.random.uniform(low=self.x_min + 1, high=self.x_max - 1)
            self.y = np.random.uniform(low=self.y_min + 1, high=self.y_max - 1)
            self.z = np.random.uniform(low=self.z_min + 1, high=self.z_max - 1)
        else:
            self.x = self.x_min
            self.y = self.y_min
            self.z = self.z_min
        self.generate_ref_pos_trajectory(self.ref_pos_amplitude, self.ref_pos_period, self.ref_pos_bias_a, self.ref_pos_bias_phase)
    
    def generate_ref_pos(self, _pos: np.ndarray):
        N = int(self.time_max / self.dt) + 1
        self.eta_d_all = np.tile(_pos, (N, 1))
        self.dot_eta_d_all = np.zeros((N, 3))
        self.dot2_eta_d_all = np.zeros((N, 3))
        self.psi_d_all = np.zeros(N)
        self.dot_psi_d_all = np.zeros(N)
        self.dot2_psi_d_all = np.zeros(N)
    
    def generate_random_ref_pos(self, is_random: bool, random_pos0: bool, outer_param=None):
        if outer_param is not None:
            pos = np.array(outer_param[0])
        else:
            if is_random:
                pos = np.array([
                    np.random.uniform(self.x_min + 1, self.x_max - 1),
                    np.random.uniform(self.y_min + 1, self.y_max - 1),
                    np.random.uniform(self.z_min + 1, self.z_max - 1)
                ])
            else:
                pos = np.array([self.x_max, self.y_max, self.z_max])
        if random_pos0:
            self.x = np.random.uniform(low=self.x_min + 1, high=self.x_max - 1)
            self.y = np.random.uniform(low=self.y_min + 1, high=self.y_max - 1)
            self.z = np.random.uniform(low=self.z_min + 1, high=self.z_max - 1)
        else:
            self.x = self.x_min
            self.y = self.y_min
            self.z = self.z_min
        self.generate_ref_pos(pos)
    
    def controller_reset(self):
        self.ref_pos_amplitude = np.zeros(3)
        self.ref_pos_period = np.zeros(3)
        self.ref_pos_bias_a = np.zeros(3)
        self.ref_pos_bias_phase = np.zeros(3)
        self.eta_d_all = np.atleast_2d([])
        self.dot_eta_d_all = np.atleast_2d([])
        self.dot2_eta_d_all = np.atleast_2d([])
        self.psi_d_all = np.array([])
        self.dot_psi_d_all = np.array([])
        self.dot2_psi_d_all = np.array([])
        self.att_ctrl.fntsmc_reset()
        self.pos_ctrl.fntsmc_reset()
    
    def controller_reset_with_new_param(self, new_pos_param: fntsmc_param = None):
        self.ref_pos_amplitude = None
        self.ref_pos_period = None
        self.ref_pos_bias_a = None
        self.ref_pos_bias_phase = None
        self.eta_d_all = np.atleast_2d([])
        self.dot_eta_d_all = np.atleast_2d([])
        self.dot2_eta_d_all = np.atleast_2d([])
        self.psi_d_all = np.array([])
        self.dot_psi_d_all = np.array([])
        self.dot2_psi_d_all = np.array([])
        '''参考轨迹记录'''
        if new_pos_param is not None:
            self.pos_ctrl.fntsmc_reset_with_new_param(new_pos_param)
    
    def collector_reset(self):
        self.collector.reset()
