import numpy as np
import matplotlib.pyplot as plt
import utils
from mpl_toolkits.mplot3d import Axes3D
import utils
# from imag3D import utils_3d
# from scipy.spatial import transform as tra


###################### FUNCIONES ######################
def pintar(data_degrees, dif_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_degrees[:, 0], data_degrees[:, 1], dif_idx)


def ang_min(data_a_scan, data_degrees, threshold, rango):
    idx_thr = utils.first_thr_cross(data_a_scan, rango, threshold, 20, axis=-1)
    t_idx = idx_thr[:, :, 0]  # obtengo los índices de corte de umbral

    mx_idx = np.max(t_idx, axis=1)
    mn_idx = np.min(t_idx, axis=1)

    dif_idx = np.abs(mx_idx - mn_idx)  # estimando error
    idx_min = np.argmin(dif_idx, axis=0)
    angmin = data_degrees[idx_min, :]

    return angmin, dif_idx


def get_list_of_positions(n, dz, rx, ry):
    dz_range = np.random.uniform(dz[0], dz[1], n)
    rx_range = np.random.uniform(rx[0], rx[1], n)
    ry_range = np.random.uniform(ry[0], ry[1], n)
    pose_combinations = list(zip(dz_range, rx_range, ry_range))
    return pose_combinations


def get_list_of_positions2(n, dz, dy, rz, ry):
    dz_range = np.random.uniform(dz[0], dz[1], n)
    dy_range = np.random.uniform(dy[0], dy[1], n)
    ry_range = np.random.uniform(ry[0], ry[1], n)
    rz_range = np.random.uniform(rz[0], rz[1], n)
    pose_combinations = list(zip(dz_range, dy_range, rz_range, ry_range))
    return pose_combinations


def get_list_of_positions3(n, dx, dy, dz):
    dx_range = np.random.uniform(dx[0], dx[1], n)
    dy_range = np.random.uniform(dy[0], dy[1], n)
    dz_range = np.random.uniform(dz[0], dz[1], n)
    pose_combinations = list(zip(dx_range, dy_range, dz_range))
    return pose_combinations


def filter_list(pose_combinations, z_min, ang_max):
    pose_combinations2 = []
    for pose_i in pose_combinations:
        if pose_i[0] <= z_min:
            if pose_i[1] <= ang_max and pose_i[2] <= ang_max:
                pose_combinations2.append(pose_i)
        else:
            pose_combinations2.append(pose_i)
    return pose_combinations2


def filter_list2(pose_combinations, z_min, ang_max, anglim1, anglim2):
    pose_combinations2 = []
    for pose_i in pose_combinations:
        if pose_i[0] <= z_min:
            if pose_i[3] <= ang_max:
                pose_combinations2.append(pose_i)
        else:
            pose_combinations2.append(pose_i)
    pose_combinations3 = []
    for pose_i in pose_combinations2:
        if pose_i[2] >= anglim1:
            if pose_i[3] <= anglim2:
                pose_combinations3.append(pose_i)
        else:
            pose_combinations3.append(pose_i)
    return pose_combinations3


def sweep_z_rx_ry(pose, t=10):
    pass

###################### ---------- ######################

###################### FUNCIONES  ARRAY 8X16 ######################

def get_positions(shape):
    if shape == 'plane':
        pose_combinations = [(0, 0, -27, -10, 0, 0),
                              (0, 0, -27, 0, 11, 0),
                              (0, 0, -26, 8, 8, 0),
                              (0, 0, -26, 15, -8, 0),
                              (0, 0, -26, -3, -6, 0),
                              (0, 0, -26, -11, 10, 0),
                              (0, 0, -27, 8, 4, 0),
                              (0, 0, -25, -6, 9, 0),
                              (0, 0, -27, 0, 10, 0),
                              (0, 0, -27, -6, 4, 0),
                              (0, 0, -27, -6, -5, 0),
                              (0, 0, -27, 2, 13, 0),
                              (0, 0, -27, -5, -5, 0),
                              (0, 0, -28, 0, 3, 0),
                              (0, 0, -27, -10, 11, 0)]
    elif shape == 'cyl1':
        pose_combinations = [(0, 0, -40, 0, 0, 0),
                             (0, 0.9, -44, 0, 11, 10),
                             (0, 5.7, -43, 0, 12, 20),
                             (0, 12, -44, 0, 9, 30),
                             (0, 26, -46, 0, 16, 35),
                             (0, 3.5, -43, 0, 11, 5),
                             (10, 1.4, -46, 0, 18, 0),
                             (0, 3.8, -47, 0, 15, 15),
                             (-24, 6.3, -34, 0, -12, 45),
                             (-20, 23.2, -44, 0, 14, 60),
                             (0, 10, -42, 0, 5, 30),
                             (10, 16.6, -53, 0, 20, 25),
                             (0, 3.3, -43, 0, 9, 0),
                             (-30, 38.0, -49, 0, 16, 80),
                             (-10, 25.3, -43, 0, 1, 50)]

    elif shape == 'cyl2':
        pose_combinations = [(0, 0, -67, 0, 0, 0),
                             (0, 0, -67, 0, 12, 0),
                             (0, 0.9, -66, 0, 10, 10),
                             (0, 8.9, -68, 0, 15, 20),
                             (0, 9.5, -66, 0, 8, 30),
                             (0, 19.0, -71, 0, 16, 35),
                             (0, 0.8, -67, 0, 12, 5),
                             (0, -1.0, -71, 0, 18, 2),
                             (0, 8.0, -70, 0, 15, 15),
                             (-23, 8.0, -58, 0, -12, 45),
                             (-22, 30.0, -71, 0, 14, 60),
                             (-15, 13.0, -67, 0, 6, 30),
                             (0, -2.3, -68, 0, 9, 0),
                             (-30, 39.0, -74, 0, 17, 80),
                             (-18, 21.0, -69, 0, 5, 50)]

    elif shape == 'cyl3':
        # pose_combinations = [(0, 0, -41, 0, 0, 0),
        #                      (0, 0, -42, 0, 10, 0),
        #                      (0, 5, -40, 0, 15, 10),
        #                      (0, 11, -45, 0, 20, 20),
        #                      (-29, 5, -33, 0, -25, 17),
        #                      (0, 10, -44, 0, 16, 25),
        #                      (0, 18, -46, 0, 14, 30),
        #                      (-24, -6, -32, 0, -15, -10),
        #                      (0, 12, -43, 0, 20, 20),
        #                      (-10, 6, -41, 0, 0, 17),
        #                      (0, 12, -43, 0, 13, 13),
        #                      (0, -14, -43, 0, 15, -25),
        #                      (-30, -10, -30, 0, -18, -18),
        #                      (-10, 0, -40, 0, 0, -10),
        #                      (0, 22, -45, 0, 20, 30)]

        pose_combinations = [(0, 0, -51, 0, 0, 0), # a
                             (0, 0, -50, 0, 0, 0), # a
                             (0, 0, -48, 0, 0, 0), #a
                             (0, 10, -47, 0, 0, 0),
                             (0, 10, -46, 0, 0, 0), # a
                             (0, 10, -48, 0, 8, 0),
                             (0, 22, -48, 0, 16, 25),
                             (0, 22, -51, 0, 14, 20),
                             (-11, 10, -42, 0, 8, 10)]

        # pose_combinations = [(60, 0, -40, 0, 0, 0),
        #                      (60, 0, -45, 0, 0, 0),
        #                      (60, 0, -55, 0, 0, 0),
        #                      (60, 0, -60, 0, 10, 0),
        #                      (60, 0, -65, 0, 0, 0),
        #                      (60, 0, -70, 0, 0, 0),
        #                      (60, 0, -75, 0, 0, 0),
        #                      (60, 0, -72, 0, 10, 0),
        #                      (70, 0, -69, 0, 0, 0)]

    return pose_combinations