import sys
sys.path.append('./..')
import numpy as np
import matplotlib.pyplot as plt
from function import *
from data import k6, dahe, k6_2
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(linewidth=200)
du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)

# 大和机针高度9.5 行程33
# 杰克机针高度9 行程33.4


def PrintNeedleHeight(machine):
    print(np.max(machine.p1_f1[:, 2]) - machine.tph)


def threadLength(thread):
    thread = thread[:-1] - thread[1:]
    thread = modulus(thread)
    thread = np.sum(thread, axis=0).reshape(-1, 1)
    return thread


def threeThreadLength(machine):
    machine_p2_i1 = np.tile(machine.p2_i1, machine.n.shape)
    machine_p2_i2 = np.tile(machine.p2_i2, machine.n.shape)
    machine_p2_i3 = np.tile(machine.p2_i3, machine.n.shape)
    machine_p2_j1 = np.tile(machine.p2_j1, machine.n.shape)
    machine_p2_j2 = np.tile(machine.p2_j2, machine.n.shape)
    machine_p2_j3 = np.tile(machine.p2_j3, machine.n.shape)
    machine_p1_h1 = np.tile(machine.p1_h1, machine.n.shape)
    machine_p1_h2 = np.tile(machine.p1_h2, machine.n.shape)
    machine_p1_h3 = np.tile(machine.p1_h3, machine.n.shape)
    # 过线板-挑线杆-小过线杆
    machine_l_thread = np.array([machine_p2_i1, machine.p2_g1, machine.p2_f1, machine_p2_j1])
    machine_m_thread = np.array([machine_p2_i2, machine.p2_g2, machine.p2_f2, machine_p2_j2])
    machine_r_thread = np.array([machine_p2_i3, machine.p2_g3, machine.p2_f3, machine_p2_j3])
    # 针夹头孔-过线板-挑线杆-小过线杆
    # machine_l_thread = np.array([machine.p1_g1, machine_p2_i1, machine.p2_g1, machine.p2_f1, machine_p2_j1])
    # machine_m_thread = np.array([machine.p1_g2, machine_p2_i2, machine.p2_g2, machine.p2_f2, machine_p2_j2])
    # machine_r_thread = np.array([machine.p1_g3, machine_p2_i3, machine.p2_g3, machine.p2_f3, machine_p2_j3])
    # 针板孔-针夹头孔-过线板-挑线杆-小过线杆
    # machine_l_thread = np.array([machine_p1_h1, machine.p1_g1, machine_p2_i1, machine.p2_g1, machine.p2_f1, machine_p2_j1])
    # machine_m_thread = np.array([machine_p1_h2, machine.p1_g2, machine_p2_i2, machine.p2_g2, machine.p2_f2, machine_p2_j2])
    # machine_r_thread = np.array([machine_p1_h3, machine.p1_g3, machine_p2_i3, machine.p2_g3, machine.p2_f3, machine_p2_j3])
    machine_left_thread = threadLength(machine_l_thread)
    machine_middle_thread = threadLength(machine_m_thread)
    machine_right_thread = threadLength(machine_r_thread)
    return machine_left_thread, machine_middle_thread, machine_right_thread


def threadVarietySolve(machine):
    left_thread, middle_thread, right_thread = threeThreadLength(machine)
    left_thread = left_thread - left_thread[0]
    middle_thread = middle_thread - middle_thread[0]
    right_thread = right_thread - right_thread[0]
    return left_thread, middle_thread, right_thread


def diffSolve(dahe_thread_variety, k6_thread_variety):
    dahe_left_thread_variety = dahe_thread_variety[0]
    dahe_middle_thread_variety = dahe_thread_variety[1]
    dahe_right_thread_variety = dahe_thread_variety[2]
    k6_left_thread_variety = k6_thread_variety[0]
    k6_middle_thread_variety = k6_thread_variety[1]
    k6_right_thread_variety = k6_thread_variety[2]

    diff_left = k6_left_thread_variety - dahe_left_thread_variety
    diff_middle = k6_middle_thread_variety - dahe_middle_thread_variety
    diff_right = k6_right_thread_variety - dahe_right_thread_variety
    return diff_left, diff_middle, diff_right


def plot2():
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(k6.p2_g1[:, 0], k6.p2_g1[:, 2], 'b', lw=0.7, label='杰克')
    ax1.plot(dahe.p2_g1[:, 0], dahe.p2_g1[:, 2], 'g', lw=0.7, label='大和')
    ax1.invert_xaxis()
    ax1.grid()
    ax1.legend()
    ax1.set_title('挑线凸轮4.5->4.76')
    # ax1.set_aspect(1)

    ax2 = fig.add_subplot(3, 1, 2)
    k6_velocity = np.diff(k6.theta2_0dx, axis=0) * k6.du
    dahe_velocity = np.diff(dahe.theta2_0dx, axis=0) * dahe.du
    ax2.plot(k6_velocity, 'b', lw=0.7, label='杰克')
    ax2.plot(dahe_velocity, 'g', lw=0.7, label='大和')
    ax2.grid()
    ax2.legend()
    ax2.set_title('挑线杆摆动角速度')

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(k6_velocity - dahe_velocity, lw=0.7, label=f'{np.max(np.abs(k6_velocity - dahe_velocity))}')
    ax3.grid()
    ax3.legend()
    ax3.set_title('角速度差值')

    plt.show()


def plot1(dahe_thread_variety, k6_thread_variety, diff_thread, k6_2_thread_variety):
    dahe_left_thread_variety = dahe_thread_variety[0]
    dahe_middle_thread_variety = dahe_thread_variety[1]
    dahe_right_thread_variety = dahe_thread_variety[2]
    k6_left_thread_variety = k6_thread_variety[0]
    k6_middle_thread_variety = k6_thread_variety[1]
    k6_right_thread_variety = k6_thread_variety[2]
    diff_left = diff_thread[0]
    diff_middle = diff_thread[1]
    diff_right = diff_thread[2]

    k6_2_left_thread_variety = k6_2_thread_variety[0]
    k6_2_middle_thread_variety = k6_2_thread_variety[1]
    k6_2_right_thread_variety = k6_2_thread_variety[2]
    diff_left = k6_left_thread_variety - k6_2_left_thread_variety
    diff_middle = k6_middle_thread_variety - k6_2_middle_thread_variety
    diff_right = k6_right_thread_variety - k6_2_right_thread_variety

    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(k6_left_thread_variety, '-r', lw=0.5, label='K6(105左置)左')
    ax1.plot(k6_middle_thread_variety, '--r', lw=0.5, label='K6(105左置)中')
    ax1.plot(k6_right_thread_variety, ':r', lw=0.5, label='K6(105左置)右')
    # ax1.plot(dahe_left_thread_variety, '-g', lw=0.5, label='大和左')
    # ax1.plot(dahe_middle_thread_variety, '--g', lw=0.5, label='大和中')
    # ax1.plot(dahe_right_thread_variety, ':g', lw=0.5, label='大和右')
    ax1.plot(k6_2_left_thread_variety, '-b', lw=0.5, label='K6(116右置)左')
    ax1.plot(k6_2_middle_thread_variety, '--b', lw=0.5, label='K6(116右置)中')
    ax1.plot(k6_2_right_thread_variety, ':b', lw=0.5, label='K6(116右置)右')
    ax1.legend()
    ax1.grid()
# 挑线凸轮4.5->4.78\n\
# 针杆过线板、火柴杆改成和大和一样\n\
# 火柴杆43->46.67 | 33->32.98 | 23->21.54
#     ax1.set_title(
#         "过线板-挑线杆-小过线杆,杰克、大和长度变化量\n\
# 挑线轴位置105, 左置\n"
#     )
    # ax1.set_title("针夹头孔-过线板-挑线杆-小过线杆,杰克、大和长度变化量")
    # ax1.set_title("针板孔-针夹头孔-过线板-挑线杆-小过线杆,杰克、大和长度变化量")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(diff_left, lw=1, label=f'左 {str(np.max(np.abs(diff_left)))}')
    ax2.plot(diff_middle, '--', lw=1, label=f'中 {str(np.max(np.abs(diff_middle)))}')
    ax2.plot(diff_right, ':', lw=1, label=f'右 {str(np.max(np.abs(diff_right)))}')
    ax2.legend()
    ax2.grid()
    ax2.set_title("差值(杰克105左置-杰克116右置)")

    plt.show()


def k6Solve(k6):
    def changedParams():
        k6.p2_d = np.array([-105, -0.7, -31])  # 挑线轴轴心
        # k6.p2_f1_ = np.array([48, 40.29809725, 28])  # 挑线杆短孔位置
        # k6.p2_g1_ = np.array([99, 40.29809725, 23.5])  # 挑线杆长孔位置

        # k6.p2_i1 = dahe.p2_i1
        # k6.p2_i2 = dahe.p2_i2
        # k6.p2_i3 = dahe.p2_i3
        # k6.p2_i1 = np.array([5, 31.98601031, -38.60759007])
        # k6.p2_i2 = np.array([0, 31.98601031, -38.60759007])
        # k6.p2_i3 = np.array([-5, 31.98601031, -38.60759007])

        # k6.p2_d = dahe.p2_d
        # k6.l2_ab = dahe.l2_ab
        # k6.l2_bc = dahe.l2_bc
        # k6.l2_cd = dahe.l2_cd

    def optimizeParams():
        # k6.p2_j1 = np.array([-124.53347593, 46.5148213, 46.67996653])  # 火柴杆左
        # k6.p2_j2 = np.array([-118.9043108, 49.7648213, 32.98980179])  # 火柴杆中
        # k6.p2_j3 = np.array([-113.27514568, 53.0148213, 21.54969446])  # 火柴杆右
        k6.p2_j1 = dahe.p2_j1
        k6.p2_j2 = dahe.p2_j2
        k6.p2_j3 = dahe.p2_j3

        # k6.l2_ab = 4.78495427  # 挑线偏心
        # k6.theta2_cdx = 8.71332088 * k6.hd  # 挑线杆与挑线摆杆角度
        # k6.p2_d = np.array([-104, -1.2, -31])  # 挑线轴轴心

    changedParams()
    # optimizeParams()
    k6.Solve()
    return threadVarietySolve(k6)


def optimizeFunction():
    def func(x0):
        k6.p2_j1 = np.array([-124.53347593, 46.5148213, x0[0]])  # 火柴杆左
        k6.p2_j2 = np.array([-118.9043108, 49.7648213, x0[1]])  # 火柴杆中
        k6.p2_j3 = np.array([-113.27514568, 53.0148213, x0[2]])  # 火柴杆右
        # k6.l2_ab = x0[3]  # 挑线偏心
        # k6.theta2_cdx = x0[4] * k6.hd  # 挑线杆与挑线摆杆角度
        # k6.p2_d = np.array([-104, x0[5], -31])  # 挑线轴轴心
        k6_thread_variety = k6Solve(k6)
        diff_thread = diffSolve(dahe_thread_variety, k6_thread_variety)
        return np.sum(diff_thread[0]**2) + np.sum(diff_thread[1]**2) + np.sum(diff_thread[2]**2)

    x0 = [43, 33, 23, 4.5, 10, 1.32]
    result = minimize(fun=func, x0=x0)
    print(result)
    print(result.x)


dahe_thread_variety = threadVarietySolve(dahe)
k6_thread_variety = k6Solve(k6)
diff_thread = diffSolve(dahe_thread_variety, k6_thread_variety)
k6_2_thread_variety = threadVarietySolve(k6_2)

plot1(dahe_thread_variety, k6_thread_variety, diff_thread, k6_2_thread_variety)
# plot2()
# optimizeFunction()

# k6.theta2_0dx[k6.theta2_0dx < (-np.pi)] = 2 * np.pi + k6.theta2_0dx[k6.theta2_0dx < (-np.pi)]
# print(np.min(k6.theta2_0dx[k6.theta2_0dx < 0]) * du)
# plt.plot(k6.theta2_0dx * du)
# plt.show()
