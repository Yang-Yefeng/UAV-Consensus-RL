import numpy as np
from environment.envs.UAV.uav_att_ctrl import uav_att_ctrl, fntsmc_param, uav_param


# att = uav_att_ctrl(uav_param(), fntsmc_param())
# ref_att_amplitude = np.array([2, 2, 2])
# ref_att_period = np.array([1, 1, 1])
# ref_att_bias_a = np.array([0, 0, 0])
# ref_att_bias_phase = np.array([0, 0, 0])
# r, dr, ddr = att.generate_ref_att_trajectory(ref_att_amplitude, ref_att_period, ref_att_bias_a, ref_att_bias_phase)
# print(r.shape, dr.shape, ddr.shape)

a= np.concatenate((np.random.uniform(low=0, high=3)*np.ones(3), [np.random.uniform(low=0, high=1)]))
print(a)