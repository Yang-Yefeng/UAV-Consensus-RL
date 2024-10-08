import numpy as np


def deg2rad(deg):
    return deg * np.pi / 180.


def rad2deg(deg):
    return deg * 180. / np.pi


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def T(x):
    return np.tan(x)


def sind(theta):
    return np.sin(theta / 180.0 * np.pi)


def cosd(theta):
    return np.cos(theta / 180.0 * np.pi)
