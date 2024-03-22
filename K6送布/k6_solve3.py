# 差动降本
import sys
sys.path.append('./..')
import numpy as np
import matplotlib.pyplot as plt
from function import *
from k6_data import *
from dahe_data import *
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import axes3d

np.set_printoptions(linewidth=200)
du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)


def shifting(jack):
    jack.p4_l1 = jack.p4_l1 - jack.p4_l1[0] + dahe.p4_l1[0]
    jack.p4_r1 = jack.p4_r1 - jack.p4_r1[0] + dahe.p4_r1[0]


def plot1(dahe, jack, n):
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_aspect(1)

    dahe.p4_r1 = dahe.p4_r1 + np.array([0, 0, +0.02])
    dahe_line1 = ax1.plot(dahe.p4_r1[:, 1], dahe.p4_r1[:, 2], 'g', lw=1, label="大和")
    jack_line1 = ax1.plot(jack.p4_r1[:, 1], jack.p4_r1[:, 2], 'b', lw=1, label="杰克")

    ax1.legend()
    plt.show()


def fitDahe(jack):  # 差动牙行程调成和大和一样
    dahe_r1_trajectory = np.ptp(dahe.p4_r1[:, 1])

    def optimizeFunction(x0):
        jack.solveTrajectory(-13, x0[0])
        return (np.ptp(jack.p4_r1[:, 1]) - dahe_r1_trajectory)**2

    x0 = [60]
    result = minimize(optimizeFunction, x0=x0)
    return result.x[0]


dahe = Dahe()
dahe.solveTrajectory(-17.3, -5)  # 针距 小6.2 ~ 大-20.3,差动 小-9.6 ~ 大15.5

# n = np.array([[62.25271918 * hd]])
# jack1 = JackV1()
# jack1.n = n
# jack1.p4_x = np.array([0, 20, -53])
# jack1.l4_xw = 51.5
# jack1.l4_wv = 28
# jack1.l4_ts = 23  # 23,15
# jack1.theta4_uts = -143.12684762 * hd  # -143.12684762,-130.15159757
# jack1.theta4_efg2 = -30.5 * hd  # -33.8 * hd
# jack1.solveTrajectory(-13, 68)  # 针距 小12 ~ 大-13,差动 小48 ~ 大68
# jack_p4_l1_above = jack2.p4_l1[jack2.p4_l1[:, 2] > 38.8]
# jack_p4_r1_above = jack2.p4_r1[jack2.p4_r1[:, 2] > 38.8]
# print("大和|针距", np.ptp(dahe.p4_l1[:, 1]), " 差动", np.ptp(dahe.p4_r1[:, 1]))
# print("杰克1|针距", np.ptp(jack1.p4_l1[:, 1]), " 差动", np.ptp(jack1.p4_r1[:, 1]), "差动比", np.ptp(jack1.p4_r1[:, 1]) / np.ptp(jack1.p4_l1[:, 1]))
# print("杰克|针距", np.ptp(jack_p4_l1_above[:, 1]), " 差动", np.ptp(jack_p4_r1_above[:, 1]), " 差动比", np.ptp(jack_p4_r1_above[:, 1]) / np.ptp(jack_p4_l1_above[:, 1]))
# print(jack1.p4_l1, jack1.p4_r1)
# fitDahe(jack1)
# shifting(jack1)
# plot1(dahe, jack1, n)


# 调整角度
# n = np.array([[54.9997273 * hd]])
jack2 = JackV1()
jack2.n = n
jack2.p4_x = np.array([0, 20, -53])
jack2.p4_l1_ = np.array([0, -16.7, 39.75])  # 送布牙齿坐标1
jack2.p4_r1_ = np.array([0, 7.25, 28.25])  # 差动牙齿坐标1
jack2.l4_xw = 51.5  # 51.5
jack2.l4_wv = 28  # 28
jack2.l4_ut = 27.5
jack2.l4_ts = 20  # 28,15
jack2.l4_sh2 = 28  # 28
jack2.theta4_efg = 138.7 * hd
jack2.theta4_efg2 = -30.8 * hd
jack2.theta4_uts = (44.00968115 - 180) * hd
jack2.solveTrajectory(-13, 58)  # 针距 小12 ~ 大-13,差动 小48 ~ 大68
# print("大和|针距", np.ptp(dahe.p4_l1[:, 1]), " 差动", np.ptp(dahe.p4_r1[:, 1]))
print("杰克|针距", np.ptp(jack2.p4_l1[:, 1]), " 差动", np.ptp(jack2.p4_r1[:, 1]), " 差动比", np.ptp(jack2.p4_r1[:, 1]) / np.ptp(jack2.p4_l1[:, 1]))
# print(jack2.p4_l1, jack2.p4_r1)
# fitDahe(jack2)
# shifting(jack2)
# plot1(dahe, jack2, n)


# n = np.array([[82.5667238 * hd]])
# jack3 = JackV1()
# jack3.n = n
# jack3.p4_x = np.array([0, 20, -53])
# jack3.p4_l1_ = np.array([0, -16.7, 39.75])  # 送布牙齿坐标1
# jack3.p4_r1_ = np.array([0, 8.25, 28.25])  # 差动牙齿坐标1
# jack3.l4_xw = 51.5  # 51.5
# jack3.l4_wv = 20  # 28
# jack3.l4_ut = 27.5
# jack3.l4_ts = 15  # 28,15
# jack3.l4_sh2 = 26  # 28
# jack3.theta4_efg = 138.7 * hd
# jack3.theta4_efg2 = -30.8 * hd
# jack3.theta4_uts = (80 - 180) * hd
# jack3.solveTrajectory(-13, 58)  # 针距 小12 ~ 大-13,差动 小41 ~ 大61
# print("大和|针距", np.ptp(dahe.p4_l1[:, 1]), " 差动", np.ptp(dahe.p4_r1[:, 1]))
# print("杰克|针距", np.ptp(jack3.p4_l1[:, 1]), " 差动", np.ptp(jack3.p4_r1[:, 1]), " 差动比", np.ptp(jack3.p4_r1[:, 1]) / np.ptp(jack3.p4_l1[:, 1]))
# print(jack3.p4_l1, jack3.p4_r1)
# fitDahe(jack3)
# shifting(jack3)
# plot1(dahe, jack3, n)
