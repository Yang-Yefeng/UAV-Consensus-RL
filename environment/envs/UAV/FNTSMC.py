import numpy as np


class fntsmc_param:
	def __init__(self,
				 k1: np.ndarray = np.zeros(3),
				 k2: np.ndarray = np.zeros(3),
				 k3: np.ndarray = np.zeros(3),
				 k4: np.ndarray = np.zeros(3),
				 alpha1: np.ndarray = 1.01 * np.ones(3),
				 alpha2: np.ndarray = 1.01 * np.ones(3),
				 dim: int = 3,
				 dt: float = 0.01
				 ):
		self.k1 = k1
		self.k2 = k2
		self.k3 = k3
		self.k4 = k4
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		self.dim = dim
		self.dt = dt


class fntsmc:
	def __init__(self,
				 param: fntsmc_param = None,
				 k1: np.ndarray = np.array([0.3, 0.3, 1.]),
				 k2: np.ndarray = np.array([0.5, 0.5, 1.]),
				 k3: np.ndarray = np.array([0.05, 0.05, 0.05]),
				 k4: np.ndarray = np.array([6, 6, 6]),
				 alpha1: np.ndarray = np.array([1.01, 1.01, 1.01]),
				 alpha2: np.ndarray = np.array([1.01, 1.01, 1.01]),
				 dim: int = 3,
				 dt: float = 0.01):
		self.k1 = k1 if param is None else param.k1
		self.k2 = k2 if param is None else param.k2
		self.k3 = k3 if param is None else param.k3
		self.k4 = k4 if param is None else param.k4
		self.alpha1 = alpha1 if param is None else param.alpha1
		self.alpha2 = alpha2 if param is None else param.alpha2
		self.dt = dt if param is None else param.dt
		self.dim = dim if param is None else param.dim
		self.s = np.zeros(self.dim)
		self.control_in = np.zeros(self.dim)
		self.control_out = np.zeros(self.dim)

	@staticmethod
	def sig(x, a, kt=5):
		return np.fabs(x) ** a * np.tanh(kt * x)

	def control_update_inner(self,
							 e_rho: np.ndarray,
							 dot_e_rho: np.ndarray,
							 dd_ref: np.ndarray,
							 W: np.ndarray,
							 dW: np.ndarray,
							 omega: np.ndarray,
							 A_omega: np.ndarray,
							 B_omega: np.ndarray,
							 obs: np.ndarray,
							 att_only: bool = False):
		if not att_only:
			dd_ref = np.zeros(self.dim)
		self.s = dot_e_rho + self.k1 * e_rho + self.k2 * self.sig(e_rho, self.alpha1)
		tau1 = np.dot(W, A_omega) + np.dot(dW, omega) - dd_ref
		tau2 = (self.k1 + self.k2 * self.alpha1 * self.sig(e_rho, self.alpha1 - 1)) * dot_e_rho
		tau3 = obs + self.k3 * np.tanh(5 * self.s) + self.k4 * self.sig(self.s, self.alpha2)
		self.control_in = -np.dot(np.linalg.inv(np.dot(W, B_omega)), tau1 + tau2 + tau3)

	def control_update_outer(self,
							 e_eta: np.ndarray,
							 dot_e_eta: np.ndarray,
							 dot_eta: np.ndarray,
							 kt: float,
							 m: float,
							 dd_ref: np.ndarray,
							 obs: np.ndarray):
		self.s = dot_e_eta + self.k1 * e_eta + self.k2 * self.sig(e_eta, self.alpha1)
		u1 = -kt / m * dot_eta - dd_ref
		u2 = (self.k1 + self.k2 * self.alpha1 * self.sig(e_eta, self.alpha1 - 1)) * dot_e_eta
		u3 = obs + self.k3 * np.tanh(5 * self.s) + self.k4 * self.sig(self.s, self.alpha2)
		self.control_out = -(u1 + u2 + u3)
	
	def fntsmc_reset(self):
		self.s = np.zeros(self.dim)
	
	def fntsmc_reset_with_new_param(self, param:fntsmc_param):
		self.k1 = param.k1
		self.k2 = param.k2
		self.k3 = param.k3
		self.k4 = param.k4
		self.alpha1 = param.alpha1
		self.alpha2 = param.alpha2
		self.dim = param.dim
		self.dt = param.dt
		
		self.s = np.zeros(self.dim)
	