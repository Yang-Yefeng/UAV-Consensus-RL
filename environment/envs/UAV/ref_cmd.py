import numpy as np


def ref_inner(time, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
	"""
    :param time:        time
    :param amplitude:   amplitude
    :param period:      period
    :param bias_a:      amplitude bias
    :param bias_phase:  phase bias
    :return:            reference attitude angles and their 1st - 3rd derivatives
                        [phi_d, theta_d, psi_d]
                        [dot_phi_d, dot_theta_d, dot_psi_d]
                        [dot2_phi_d, dot2_theta_d, dot2_psi_d]
                        [dot3_phi_d, dot3_theta_d, dot3_psi_d]
    """
	w = 2 * np.pi / period
	_r = amplitude * np.sin(w * time + bias_phase) + bias_a
	_dr = amplitude * w * np.cos(w * time + bias_phase)
	_ddr = -amplitude * w ** 2 * np.sin(w * time + bias_phase)
	_dddr = -amplitude * w ** 3 * np.cos(w * time + bias_phase)
	return _r, _dr, _ddr, _dddr


def ref_uav(time: float, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
	"""
    :param time:        time
    :param amplitude:   amplitude
    :param period:      period
    :param bias_a:      amplitude bias
    :param bias_phase:  phase bias
    :return:            reference position and yaw angle and their 1st - 3rd derivatives
                        [x_d, y_d, z_d, yaw_d]
                        [dot_x_d, dot_y_d, dot_z_d, dot_yaw_d]
                        [dot2_x_d, dot2_y_d, dot2_z_d, dot2_yaw_d]
                        [dot3_x_d, dot3_y_d, dot3_z_d, dot3_yaw_d]
    """
	w = 2 * np.pi / period
	_r = amplitude * np.sin(w * time + bias_phase) + bias_a
	_dr = amplitude * w * np.cos(w * time + bias_phase)
	_ddr = -amplitude * w ** 2 * np.sin(w * time + bias_phase)
	# _dddr = -amplitude * w ** 3 * np.cos(w * time + bias_phase)
	return _r, _dr, _ddr


def offset_uav_n(time: float, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
	'''所有参数，除了时间之外，都是二维的 numpy，每一行表示一个无人机，行里面的每一列分别表示 x y z'''
	w = 2 * np.pi / period
	_off = amplitude * np.sin(w * time + bias_phase) + bias_a
	_doff = amplitude * w * np.cos(w * time + bias_phase)
	_ddoff = -amplitude * w ** 2 * np.sin(w * time + bias_phase)
	return _off, _doff, _ddoff


def generate_uncertainty(time: float, is_ideal: bool = False) -> np.ndarray:
	"""
    :param time:        time
    :param is_ideal:    ideal or not
    :return:            Fdx, Fdy, Fdz, dp, dq, dr
    """
	if is_ideal:
		return np.array([0, 0, 0, 0, 0, 0]).astype(float)
	else:
		T = 2
		w = 2 * np.pi / T
		phi0 = 0.
		if time <= 5:
			phi0 = 0.
			Fdx = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(3 * w * time + phi0) + 0.2
			Fdy = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(3 * w * time + phi0) + 0.4
			Fdz = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(3 * w * time + phi0) - 0.5
			
			dp = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(w * time + phi0)
			dq = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(w * time + phi0)
			dr = 0.5 * np.cos(w * time + phi0) + 0.2 * np.sin(w * time + phi0)
		elif 5 < time <= 10:
			Fdx = 1.5
			Fdy = 0.4 * (time - 5.0)
			Fdz = -0.6
			
			dp = 0.5 * np.sin(w * time + phi0) + 0.2 * np.cos(2 * np.sin(2 * w) * time + phi0)
			dq = 0.5 * np.cos(1.5 * np.sin(2 * w) * time + phi0) + 0.2 * np.sin(w * time + phi0)
			dr = 0.5 * np.sign(np.round(time - 5) % 2 - 0.5)
		else:
			phi0 = np.pi / 2
			Fdx = 0.5 * np.sin(np.cos(2 * w) * time + phi0) - 1.0 * np.cos(3 * np.sin(w) * time + phi0)
			Fdy = 0.5 * np.sign(np.round(time - 10) % 3 - 1.5) + 0.5 * np.sin(2 * w * time + phi0) - 0.4
			Fdz = 0.5 * np.cos(w * time + phi0) - 1.0 * np.sin(3 * w + time + phi0) + 1.0
			
			dp = 0.5 * np.sin(np.sin(2 * w) * time + phi0) + 0.2 * np.cos(w * time + phi0)
			dq = 1.5 * np.cos(w * time + phi0) + 0.2 * np.sin(w * time + phi0) - 0.7
			dr = 0.5 * np.cos(2 * w * time + phi0) + 0.6 * np.sin(w * time + phi0)
		
		return np.array([Fdx, Fdy, Fdz, dp, dq, dr])


def ref_uav_sequence(dt: float,
					 tm: float,
					 amplitude: np.ndarray,
					 period: np.ndarray,
					 bias_a: np.ndarray,
					 bias_phase: np.ndarray):
	w = 2 * np.pi / period
	N = int(np.round(tm / dt))
	_r = np.zeros((N, 4))
	_dr = np.zeros((N, 4))
	_ddr = np.zeros((N, 4))
	for i in range(N):
		_r[i, :] = amplitude * np.sin(w * i * dt + bias_phase) + bias_a
		_dr[i, :] = amplitude * w * np.cos(w * i * dt + bias_phase)
		_ddr[i, :] = -amplitude * w ** 2 * np.sin(w * i * dt + bias_phase)
	return _r, _dr, _ddr


def offset_uav_n_sequence(dt: float, tm: float, A: np.ndarray, T: np.ndarray, ba: np.ndarray, bp: np.ndarray):
	N = int(np.round(tm / dt))
	uav_num = A.shape[0]
	_off = np.zeros((N, uav_num, 3))
	_doff = np.zeros((N, uav_num, 3))
	_ddoff = np.zeros((N, uav_num, 3))
	w = 2 * np.pi / T
	for i in range(N):
		_off[i, :, :] = A * np.sin(w * i * dt + bp) + ba
		_doff[i, :, :] = A * w * np.cos(w * i * dt + bp)
		_ddoff[i, :, :] = -A * w ** 2 * np.sin(w * i * dt + bp)
	return _off, _doff, _ddoff
