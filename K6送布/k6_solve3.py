# 差动降本
import sys
sys.path.append(sys.path[0] + '\..')
import numpy as np
import matplotlib.pyplot as plt
from function import *
from k6_data import *
from dahe_data import *
from scipy.optimize import minimize

np.set_printoptions(linewidth=200)
du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)


def shifting(jack):
    jack.p4_l1 = jack.p4_l1 - jack.p4_l1[0] + dahe.p4_l1[0]
    jack.p4_r1 = jack.p4_r1 - jack.p4_r1[0] + dahe.p4_r1[0]


def fitDahe(jack, angle):  # 行程调成和大和一样
    def optimizeFunction_l(x0):
        jack.solveStroke(x0[0], jack.theta4_0xw * jack.du)
        return (np.ptp(jack.p4_l1[:, 1]) - dahe_l1_stroke)**2

    def optimizeFunction_r(x0):
        jack.solveStroke(jack.theta4_0d2d3 * jack.du, x0[0])
        return (np.ptp(jack.p4_r1[:, 1]) - dahe_r1_stroke)**2

    dahe_l1_stroke = np.ptp(dahe.p4_l1[:, 1])
    dahe_r1_stroke = np.ptp(dahe.p4_r1[:, 1])

    x0 = [-13]
    result = minimize(optimizeFunction_l, x0=x0)

    x0 = [angle]
    result = minimize(optimizeFunction_r, x0=x0)


def plot1(dahe, jack_source, jack_v1):
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    dahe.p4_r1 += np.array([0, 0, 0.04])

    ax1.plot([26, 32], [39.3, 39.3], 'k', lw=2, label="针板")
    ax1.plot(dahe.p4_r1[:, 1], dahe.p4_r1[:, 2], 'g', lw=0.5, label="大和")
    ax1.plot(jack_source.p4_r1[:, 1], jack_source.p4_r1[:, 2], 'b', lw=0.5, label="杰克(改前)")
    ax1.plot(jack_v1.p4_r1[:, 1], jack_v1.p4_r1[:, 2], 'r', lw=0.5, label="杰克(改后)")
    l_stroke = np.ptp(dahe.p4_l1[:, 1])
    r_stroke = np.ptp(dahe.p4_r1[:, 1])
    ax1.set_title("杰克与大和差动牙轨迹对比(针距{0:.1f}mm,差动{1:.1f}mm)".format(l_stroke, r_stroke))

    ax1.set_aspect(1)
    ax1.set_xlim([26, 32])
    ax1.set_ylim([38, 41])
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid()
    plt.show()


dahe = Dahe()
dahe.solveStroke(-7, 15.5)  # 针距 小6.2 ~ 大-20.3,差动 小-9.6 ~ 大15.5

jack1 = JackSource()
jack1.theta4_uts = -134.17943223 * hd  # -147.154682283,-134.17943223
jack1.solveStroke(-13, -5)  # 针距 小12 ~ 大-13,差动 小-25 ~ 大-2
fitDahe(jack1, -5)
print("大和|针距", np.ptp(dahe.p4_l1[:, 1]), " 差动", np.ptp(dahe.p4_r1[:, 1]))
print("杰克1|针距", np.ptp(jack1.p4_l1[:, 1]), " 差动", np.ptp(jack1.p4_r1[:, 1]), "差动比", np.ptp(jack1.p4_r1[:, 1]) / np.ptp(jack1.p4_l1[:, 1]))
shifting(jack1)


# n = np.array([[61.8293117 * hd]])
jack2 = JackV1()
jack2.n = n
jack2.theta4_b1ab3 = -62.5 * hd
jack2.p4_x = np.array([0, 20, -53])
jack2.p4_l1_ = np.array([0, -16.7, 39.75])  # 送布牙齿坐标1
jack2.p4_r1_ = np.array([0, 7.25, 28.25])  # 差动牙齿坐标1
jack2.l4_xw = 51.5  # 51.5
jack2.l4_wv = 28  # 28
jack2.l4_ut = 27.5
jack2.l4_ts = 15  # 27,15
jack2.l4_sh2 = 28  # 28
jack2.theta4_efg = 138.7 * hd
jack2.theta4_efg2 = -30.8 * hd
jack2.theta4_uts = (52.78785217 - 180) * hd
jack2.solveStroke(-13, 68)  # 针距 小12 ~ 大-13,差动 小48 ~ 大68
fitDahe(jack2, 60)
# print("大和|针距", np.ptp(dahe.p4_l1[:, 1]), " 差动", np.ptp(dahe.p4_r1[:, 1]))
print("杰克|针距", np.ptp(jack2.p4_l1[:, 1]), " 差动", np.ptp(jack2.p4_r1[:, 1]), " 差动比", np.ptp(jack2.p4_r1[:, 1]) / np.ptp(jack2.p4_l1[:, 1]))
# print(jack2.p4_l1, jack2.p4_r1)
shifting(jack2)

plot1(dahe, jack1, jack2)
