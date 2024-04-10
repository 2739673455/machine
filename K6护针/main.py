import sys
sys.path.append(sys.path[0] + '\..')
import numpy as np
import matplotlib.pyplot as plt
from function import *
from k6_data import k6
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation

np.set_printoptions(linewidth=200)
du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)


def NeedelHeight(machine):  # 求机针高度
    print(np.max(machine.p1_f1[:, 2] - machine.l_tph))


def SynchronousAngle(machine):  # 求 对弯针同步时(机针从最低点抬高一段距离后) 主轴正反转角度
    machine_l1_ac_1 = machine.l1_ab + machine.l1_bc - 9  # 正转高9mm
    machine_l1_ac_2 = machine.l1_ab + machine.l1_bc - 9.8  # 反转高9.8mm
    synchronous_angle1 = np.arccos((machine.l1_ab**2 + machine_l1_ac_1**2 - machine.l1_bc**2) / (2 * machine.l1_ab * machine_l1_ac_1))  # 正转角度
    synchronous_angle2 = np.arccos((machine.l1_ab**2 + machine_l1_ac_2**2 - machine.l1_bc**2) / (2 * machine.l1_ab * machine_l1_ac_2))  # 反转角度
    return np.array([[synchronous_angle1 + np.pi, -synchronous_angle2 + np.pi]]).T


def LooperPositionDiff(x, machine):  # 求 两个角度下弯针水平距离
    machine.theta3_13 = x[0] * hd
    machine.Solve()
    return (machine.p3_k[0][0] - machine.p3_k[1][0])**2


def LooperSynchronous(machine):  # 求 对弯针机针同步时 主轴与上轴角度
    angle = SynchronousAngle(machine)
    machine.n = angle
    x = [machine.theta3_13]
    result = minimize(fun=LooperPositionDiff, x0=x, args=machine, tol=1e-15)
    print(result)
    print(result.x)


def InvLooperSolve(machine, p3_k_x, i):  # 弯针尖水平位置->上主轴角度 i=±1
    p3_i_x = p3_k_x - machine.p3_k_[0]
    theta3_0hi = np.arccos((p3_i_x - machine.p3_h[0]) / machine.l3_hi)
    theta3_0gf = theta3_0hi - machine.theta3_fhi
    p3_f = p_rl(machine.p3_g, machine.l3_fg, theta3_0gf, 1)
    l3_ef_yz = (np.sqrt(machine.l3_ef**2 - (machine.p3_a[0] - p3_f[:, 0])**2))
    _, theta3_0ae = rrr(p3_f[:, 1:3], machine.p3_a[[1, 2]], l3_ef_yz, machine.l3_ae, i)
    theta1_ab = theta3_0ae - machine.theta3_13 - machine.theta1_0ab
    return theta1_ab


def Plot1(angle2, distance_source, distance_source1, distance_v1, distance_v11):
    # def update(i):
    #     ax1.cla()
    #     ax1.set_xlim(-8, 8)
    #     ax1.set_ylim(-200, -180)
    #     ax1.plot([k6.p5_e2[i, 1], k6.p5_f2[i, 1]], [k6.p5_e2[i, 2], k6.p5_f2[i, 2]], 'r')
    #     ax1.plot([-0.4, -0.4], [-200, -180], 'k')
    #     ax1.text(0, -190, "{:.4f}".format(distance_source1[i]))
    #     ax1.text(0, -189, i)

    #     ax2.cla()
    #     ax2.plot(distance_source1, 'b')
    #     ax2.plot(i, distance_source1[i], 'o', c='k')
    #     ax2.plot(angle2[0] * du, distance_source[0], '.r')
    #     ax2.plot(angle2[1] * du, distance_source[1], '.r')
    #     ax2.text(angle2[0] * du - 120, distance_source[0], distance_source[0].astype(np.float16))
    #     ax2.text(angle2[1] * du - 120, distance_source[1], distance_source[1].astype(np.float16))

    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ani = FuncAnimation(fig, update, frames=360, interval=5, repeat=True)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(distance_source1, '--c')
    ax1.scatter(angle2[0] * du, distance_source[0], c='y', s=20)
    ax1.scatter(angle2[1] * du, distance_source[1], c='y', s=20)
    ax1.text(angle2[0] * du - 80, distance_source[0], distance_source[0].astype(np.float16))
    ax1.text(angle2[1] * du - 80, distance_source[1], distance_source[1].astype(np.float16))
    ax1.plot(distance_v11, 'b')
    ax1.scatter(angle2[0] * du, distance_v1[0], c='r', s=20)
    ax1.scatter(angle2[1] * du, distance_v1[1], c='r', s=20)
    ax1.text(angle2[0] * du + 10, distance_v1[0], distance_v1[0].astype(np.float16))
    ax1.text(angle2[1] * du + 10, distance_v1[1], distance_v1[1].astype(np.float16))

    plt.show()


def HuZhenDistance(machine, angle):  # 两位置下护针与机针间隙
    machine.n = angle
    machine.Solve()
    huzhen_distance = np.array([- 0.4 - machine.p5_e2[0, 1], -0.4 - machine.p5_e2[1, 1]])
    return huzhen_distance


k6.theta3_13 = -3.05214526 * hd
# LooperSynchronous(k6)
angle2 = np.array([[211.34530161 * hd],
                   [235.5973203 * hd]])


def source_solve(angle2):
    distance_source = HuZhenDistance(k6, angle2)  # 0.2~0.5,-0.3
    k6.n = n
    k6.Solve()
    distance_source1 = - 0.4 - k6.p5_e2[:, 1]
    return distance_source, distance_source1


def v1_solve(angle2):
    k6.l5_ab = 0.7
    k6.theta5_15 = 60 * hd
    k6.theta5_bcd = -150.2649146 * hd
    # k6.theta5_xy = -2 * hd
    # k6.theta5_xz1 = -8 * hd
    # k6.theta5_xz2 = -22 * hd
    distance_v1 = HuZhenDistance(k6, angle2)  # 0.2~0.5,-0.3
    k6.n = n
    k6.Solve()
    distance_v11 = - 0.4 - k6.p5_e2[:, 1]
    return distance_v1, distance_v11


def function1(x):
    k6.n = n
    k6.l5_ab = x[0]
    # k6.theta5_15 = x[1] * hd
    k6.theta5_bcd = x[2] * hd
    # k6.theta5_xy = x[3] * hd
    # k6.theta5_xz1 = x[4] * hd
    point2 = HuZhenDistance(k6, angle2)
    return np.sum((point2 - np.array([0.3, -0.1]))**2)


distance_source, distance_source1 = source_solve(angle2)
distance_v1, distance_v11 = v1_solve(angle2)
Plot1(angle2, distance_source, distance_source1, distance_v1, distance_v11)

# bounds = [[None, 0.7], [None, None], [None, None], [None, -2], [None, -8]]
# x0 = [0.58, 74.97233645, -150.43342514, - 6, -12.53367695]
# result = minimize(function1, x0, bounds=bounds)
# print(result)
# print(result.x)

# theta1_ab = InvLooperSolve(k6, np.array([[-2.8, 0, 2.8]]).T, 1)
# print(theta1_ab * du + 360)

# k6.n = np.array([[152.79985617 * hd]])
# k6.Solve()
# print(k6.p1_f1)
# print(k6.p3_k)
# print(k6.p5_e2)
# print(k6.p5_f2)
