import sys
sys.path.append(sys.path[0] + '\..')
import numpy as np
from function import *
from dahe_data import Dahe
from k6_data import JackSource, JackV1
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.set_printoptions(linewidth=200)
du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)


def fitDahe(jack):  # 送布牙行程调成和大和一样
    dahe_l1_trajectory = np.ptp(dahe.p4_l1[:, 1])

    def optimizeFunction(x0):
        jack.solveStroke(x0[0], jack.theta4_0xw * jack.du)
        return (np.ptp(jack.p4_l1[:, 1]) - dahe_l1_trajectory)**2

    x0 = [-13]
    result = minimize(optimizeFunction, x0=x0)
    return result.x[0]


def Plot1(dahe, jack_v1, jack_source):
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    jack_source.p4_l1 = jack_source.p4_l1 - jack_source.p4_l1[0] + dahe.p4_l1[0] + np.array([0, 0.1, -0.15])
    jack_v1.p4_l1 = jack_v1.p4_l1 - jack_v1.p4_l1[0] + dahe.p4_l1[0] + np.array([0, 0.025, -0.01])
    ax1.plot([3, 10], [39.3, 39.3], 'k', lw=2, label="针板")
    ax1.plot(dahe.p4_l1[:, 1], dahe.p4_l1[:, 2], 'g', lw=0.5, label="大和")
    ax1.plot(jack_source.p4_l1[:, 1], jack_source.p4_l1[:, 2], 'b', lw=0.5, label="杰克(改前)")
    ax1.plot(jack_v1.p4_l1[:, 1], jack_v1.p4_l1[:, 2], 'r', lw=0.5, label="杰克(改后)")
    ax1.set_title("杰克与大和送布牙轨迹对比(针距{0:.1f}mm)".format(np.ptp(dahe.p4_l1[:, 1])))

    # jack_v1.p4_r1 = jack_v1.p4_r1 - jack_v1.p4_r1[0] + dahe.p4_r1[0] + np.array([0, 0, -0.03])
    # jack_source.p4_r1 = jack_source.p4_r1 - jack_source.p4_r1[0] + dahe.p4_r1[0] + np.array([0, 0, -0.03])
    # ax1.plot(dahe.p4_r1[:, 1], dahe.p4_r1[:, 2], 'g', lw=0.5, label="大和")
    # ax1.plot(jack_source.p4_r1[:, 1], jack_source.p4_r1[:, 2], 'b', lw=0.5, label="杰克(改前)")
    # ax1.plot(jack_v1.p4_r1[:, 1], jack_v1.p4_r1[:, 2], 'r', lw=0.5, label="杰克(改后)")
    # ax1.set_title("杰克与大和差动牙轨迹对比")  # 差动4.3mm

    # dahe.p4_l1 = dahe.p4_l1 - dahe.p4_l1[0] + jack_v1.p4_l1[0]
    # ax1.plot(dahe.p4_l1[:, 1], 'g', lw=0.5, label="大和")
    # ax1.plot(jack_source.p4_l1[:, 1], 'b', lw=0.5, label="杰克(改前)")
    # ax1.plot(jack_v1.p4_l1[:, 1], 'r', lw=0.5, label="杰克(改后)")
    # ax1.set_title("送布牙时序对比")
    ax1.set_xlim([3, 10])
    ax1.set_ylim([38, 41])
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid()
    ax1.set_aspect(1)
    plt.show()


dahe = Dahe()
dahe.solveStroke(-17.3, -5)  # 针距 小6.2 ~ 大-20.3,差动 小-9.6 ~ 大15.5

jack_source = JackSource()
jack_source.l4_ab1 = 1.2  # 主轴抬牙偏心
jack_source.theta4_0d2d3 = fitDahe(jack_source)
jack_source.solveStroke(jack_source.theta4_0d2d3, -21)  # 针距 小12 ~ 大-13,差动 小-25 ~ 大-2

jack_v1 = JackV1()
jack_v1.l4_ts = 15  # 15
jack_v1.theta4_uts = -134.17943223 * hd  # -147.154682283,-134.17943223
jack_v1.theta4_0d2d3 = fitDahe(jack_v1)
jack_v1.solveStroke(jack_v1.theta4_0d2d3, 44.5)  # 针距 小12 ~ 大-13,差动 小41 ~ 大61

# print("杰克_source|针距", np.ptp(jack_source.p4_l1[:, 1]), " 差动", np.ptp(jack_source.p4_r1[:, 1]))
# print("杰克_v1|针距", np.ptp(jack_v1.p4_l1[:, 1]), " 差动", np.ptp(jack_v1.p4_r1[:, 1]))
# print("大和|针距", np.ptp(dahe.p4_l1[:, 1]), " 差动", np.ptp(dahe.p4_r1[:, 1]))
# print("杰克_v1|差动比", np.ptp(jack_v1.p4_r1[:, 1]) / np.ptp(jack_v1.p4_l1[:, 1]))

Plot1(dahe, jack_v1, jack_source)
