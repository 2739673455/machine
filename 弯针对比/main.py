import sys
sys.path.append('./..')
import numpy as np
from data import *
from function import *
from machine import Machine
import matplotlib.pyplot as plt
from scipy.optimize import minimize

du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)


def distance1(machine, x_val):  # 勾线时弯针位置
    machine.n = machine.invLooperSolve(x_val.reshape(-1, 1), 1)
    machine.Solve()
    y_val = machine.p3_k[:, 1] + 0.2
    y_val = y_val.astype(np.float16)
    return y_val


def distance2(machine, x_val):  # 插背时弯针位置
    machine.n = machine.invLooperSolve(x_val.reshape(-1, 1), -1)
    machine.Solve()
    y_val = machine.p3_k[:, 1]
    y_val = y_val.astype(np.float16)
    return y_val


def getLooperXCoordinate_Angle(machine, angle):  # 校正开档3.2mm时弯针的x坐标、勾线时弯针距中针0mm时弯针角度
    angle = angle - machine.theta3_13 - machine.theta1_0ab
    machine.n = np.array([[angle]])
    machine.Solve()
    machine_p3_k_x = machine.p3_k_[0] - 6 - machine.p3_k[0, 0]
    print("looper x position:", machine_p3_k_x)

    def OptimizeFunction(x0, args):  # 勾线时弯针距中针0mm
        args[0].theta3_cdy = x0[0] * hd
        args[0].Solve()
        y_val = args[0].p3_k[0, 1] + 0.2
        return y_val**2

    machine.p3_k_[0] = machine_p3_k_x
    machine.n = machine.invLooperSolve(np.array([[0]]), 1)  # 贴中针
    x0 = [-161]
    result = minimize(fun=OptimizeFunction, x0=x0, args=[machine], tol=1e-20)
    angle = result.x[0]
    print("looper x angle:", angle)
    return machine_p3_k_x, angle


def looperPositionSolve(machine, x_val1, x_val2):  # 求弯针轨迹、弯针勾线与插背位置
    k6_0mm.n = n
    k6_0mm.Solve()
    return machine.p3_k, np.hstack([distance1(machine, x_val1), distance2(machine, x_val2)])


def plot1():
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    circ1_1 = plt.Circle([2.8, 0], 0.4)
    circ1_2 = plt.Circle([0, 0], 0.4)
    circ1_3 = plt.Circle([-2.8, 0], 0.4)
    ax1.add_artist(circ1_1)
    ax1.add_artist(circ1_2)
    ax1.add_artist(circ1_3)
    ax1.plot(dahe_p3_k[:, 0], dahe_p3_k[:, 1], 'g', label='大和', lw=0.5)
    ax1.plot(k6_0_p3_k[:, 0], k6_0_p3_k[:, 1], 'b', label='杰克-18.5', lw=0.5)
    ax1.plot(k6_1_p3_k[:, 0], k6_1_p3_k[:, 1], 'r', label='杰克-17', lw=0.5)
    ax1.legend()
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.set_aspect(1)

    table_h_title = ['大和', '杰克-18.5', '杰克-17']
    table_v_title = ['勾线左', '勾线中', '勾线右', '插背左', '插背中', '插背右']
    for j, v_val in enumerate(np.linspace(-6, -4, 3)):
        ax1.text(30, v_val, table_h_title[j])
    for i, h_val in enumerate(np.linspace(25, -5, 6)):
        ax1.text(h_val, -7, table_v_title[i])
        ax1.text(h_val, -6, dahe_y[i])
        ax1.text(h_val, -5, k6_0_y[i])
        ax1.text(h_val, -4, k6_1_y[i])

    plt.show()


def k6_0mm_output(x0):  # 修改弯针摆动摆杆长度，计算轨迹与勾线插背间隙
    k6_0mm.l3_cd = x0  # 弯针摆动摆杆
    _, angle = getLooperXCoordinate_Angle(k6_0mm, 270 * k6_0mm.hd)
    k6_0mm.theta3_cdy = angle * k6_0mm.hd
    k6_0mm_p3_k, k6_0mm_y = looperPositionSolve(k6_0mm, x_val1, x_val2)
    return k6_0mm_p3_k, k6_0mm_y


x_val1 = np.array([2.8, 0, -2.8])  # 勾线时弯针尖位置
x_val2 = np.array([10, 7.2, 4.4])  # 插背时弯针尖位置
dahe_p3_k, dahe_y = looperPositionSolve(dahe, x_val1, x_val2)
# k6_3mm_p3_k, k6_3mm_y = looperPositionSolve(k6_3mm, x_val1, x_val2)
# k6_0mm_p3_k, k6_0mm_y = looperPositionSolve(k6_0mm, x_val1, x_val2)
k6_0_p3_k, k6_0_y = k6_0mm_output(18.5)
k6_1_p3_k, k6_1_y = k6_0mm_output(17)
plot1()
