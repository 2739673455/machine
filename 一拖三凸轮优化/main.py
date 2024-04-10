import sys
sys.path.append(sys.path[0] + '\..')
import copy
import numpy as np
import matplotlib.pyplot as plt
from machine import *
from function import *
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
from cam_data import cam1, cam2, cam3

np.set_printoptions(linewidth=200)
du = 180 / np.pi
hd = np.pi / 180


def SolveIntersectionIndex(cam, point_e, vector_r, i=1):  # 求与凸轮交点索引,i为朝向(1为逆时针，-1为顺时针)
    def SolveIntersectionIndex0(cam, point_e, vector):
        tangency_points = cam.tangency_points - point_e  # 平移后凸轮圆弧间切点
        angle = includedAngle(vector, tangency_points)
        index = np.argwhere(angle == np.max(angle, axis=-1).reshape(-1, 1))  # 相交圆弧序号
        index = index[:, 1]
        return index

    if type(vector_r) == np.ndarray:  # 凸轮与推杆相交,vector_r为推杆方向向量
        return SolveIntersectionIndex0(cam, point_e, vector_r)
    else:  # 凸轮与摆杆相交,vector_r为摆杆半径
        r = vector_r
        distance = modulus(cam.tangency_points - point_e) - r
        index_head = (distance <= 0) * (-2) + 1  # 圆弧起始点大于r为1，小于r为-1
        if index_head.ndim == 1:  # 若圆弧起始点坐标只有一个
            index_head = np.array([index_head])
        index_tail = np.hstack([index_head[:, 1:], index_head[:, 0:1]])
        if i == 1:
            index0 = (index_head - index_tail) > 0  # 圆弧起始点在圆外，终点在圆内的索引
        else:
            index0 = (index_head - index_tail) < 0  # 圆弧起始点在圆内，终点在圆外的索引
        index1 = SolveIntersectionIndex0(cam, np.array([0, 0]), point_e)  # 凸轮中心与曲柄中心的连线与圆弧相交索引
        index0[np.sum(index0, axis=1) == 0, index1[np.sum(index0, axis=1) == 0]] = True
        order_num = np.tile(np.arange(index0.shape[-1]), (index0.shape[0], 1))
        index = order_num[index0]
        return index


def GetCurrentArcInfo(cam, index):
    if cam.arc_c.ndim == 3:
        current_arc_c = cam.arc_c[np.arange(len(cam.arc_c)), index]
    else:
        current_arc_c = cam.arc_c[index]
    current_arc_r = cam.arc_r[index]
    current_arc_t = cam.arc_t[index]
    return current_arc_c, current_arc_r, current_arc_t


def SolveIntersectionPoint(cam, current_arc_info, point_e, vector_r, i=1):  # 求与凸轮交点,i为朝向(1为逆时针，-1为顺时针)
    current_arc_c = current_arc_info[0]
    current_arc_r = current_arc_info[1]
    current_arc_t = current_arc_info[2]
    if type(vector_r) == np.ndarray:  # 凸轮与推杆相交,vector_r为推杆方向向量
        theta1, l1 = theta_l(current_arc_c - point_e)
        theta2, _ = theta_l(vector_r)
        angle = theta1 - theta2
        l3 = current_arc_r
        a3 = l1**2 - l3**2
        a2 = - 2 * l1 * np.cos(angle)
        a1 = np.tile(1, a3.shape)
        a = np.concatenate((a1, a2, a3), axis=1)
        l2 = np.array(list(map(lambda x: np.roots(x), a)))
        l2[l2 < 0] = np.inf
        l2_index = (l2 == np.min(l2, axis=-1).reshape(-1, 1))
        l2 = l2[l2_index].reshape(-1, 1)
        intersection_point = point_e + l2 * vector_r
        stroke = l2  # 行程
        cos_force_angle = np.abs((l2**2 + l3**2 - l1**2) / (2 * l2 * l3))  # 分力夹角cos值
    else:  # 凸轮与摆杆相交,vector_r为摆杆半径
        intersection_point, _ = rrr(current_arc_c, point_e, current_arc_r, vector_r, current_arc_t * i)
        stroke = modulus(intersection_point).reshape(-1, 1)  # 行程
        force_angle = includedAngle(current_arc_c - intersection_point, point_e - intersection_point)
        cos_force_angle = np.abs(np.sin(force_angle)).reshape(-1, 1)
    return intersection_point, stroke, cos_force_angle


def SolveArmOfForce(intersection_point, current_arc_info):  # 求力臂,力臂大于0时力矩为逆时针，反之力矩为顺时针
    current_arc_c = current_arc_info[0]
    current_arc_t = current_arc_info[2]
    theta0po, l1 = theta_l(-intersection_point)
    theta0pc, _ = theta_l(current_arc_c - intersection_point)
    thetacpo = theta0po - theta0pc
    arm_of_force = l1 * np.sin(thetacpo) * current_arc_t
    return arm_of_force


def SolveTorque(force_max, force_min, stroke, cos_force_angle, arm_of_force):  # 求力矩，逆时针为正，顺时针为负
    force = (force_max - force_min) / np.ptp(stroke) * (stroke - np.min(stroke)) + force_min
    force = force / cos_force_angle
    torque = force * arm_of_force
    return torque


def solveCam(cam, point_e, vector_r, i, force_max, force_min):
    cam2 = copy.deepcopy(cam)
    cam2.cam_info[:, 2] += 7 * cam2.cam_info[:, 3]
    cam2.SetCamOutline()

    rotate_angle = np.arange(360) * hd
    cam2.Rotate(-rotate_angle)
    index = SolveIntersectionIndex(cam2, point_e, vector_r, i)
    current_arc_info = GetCurrentArcInfo(cam2, index)
    intersection_point, stroke, cos_force_angle = SolveIntersectionPoint(cam2, current_arc_info, point_e, vector_r, i)
    arm_of_force = SolveArmOfForce(intersection_point, current_arc_info)
    torque = SolveTorque(force_max, force_min, stroke, cos_force_angle, arm_of_force)
    return cam2, intersection_point, torque, stroke


def solveCoordinate(cam, num0, num1, num2, r, i):
    coordinate, _ = rrr(cam.cam_info[num1, :2],
                        cam.cam_info[num2, :2],
                        cam.cam_info[num1, 3] * cam.cam_info[num1, 2] - cam.cam_info[num0, 3] * r,
                        cam.cam_info[num2, 3] * cam.cam_info[num2, 2] - cam.cam_info[num0, 3] * r,
                        i)
    return coordinate


def plot1(cam4, cam5, cam6, p_set1, p_set2, p_set3):
    def update(i):
        ax1.cla()
        ax1.set_aspect(1)
        ax1.set_xlim([-50, 50])
        ax1.set_ylim([-50, 50])
        ax1.plot(cam4.p[i, :, 0], cam4.p[i, :, 1], 'r')
        ax1.plot(cam5.p[i, :, 0], cam5.p[i, :, 1], 'b')
        ax1.plot(cam6.p[i, :, 0], cam6.p[i, :, 1], 'g')
        ax1.plot(p_set1[i, 0], p_set1[i, 1], 'or')
        ax1.plot(p_set2[i, 0], p_set2[i, 1], 'ob')
        ax1.plot(p_set3[i, 0], p_set3[i, 1], 'og')

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ani = FuncAnimation(fig, update, frames=360, interval=5, repeat=True)
    plt.show()


point_e = np.array([0.5, 0])
vector_e_t = np.array([point_e[1], point_e[0]]) / modulus(point_e)
cam4, intersection_point1, torque1, stroke1 = solveCam(cam1, point_e, vector_e_t, 1, 181, 100)
cam5, intersection_point2, torque2, stroke2 = solveCam(cam2, np.array([18.4, 11]), 21, 1, 80, 80)
cam6, intersection_point3, torque3, stroke3 = solveCam(cam3, np.array([-20.5, 21]), 23, -1, 60, 20)
torque_total = torque1 + torque2 + torque3
# plot1(cam4, cam5, cam6, intersection_point1, intersection_point2, intersection_point3)


def cam1_version1():
    def func1(x0):
        r1 = x0[0]
        r2 = x0[1]
        # print(x0)
        coordinate1 = solveCoordinate(cam, 1, 0, 2, r1, 1)
        coordinate2 = solveCoordinate(cam, 3, 2, 4, r2, 1)
        cam.cam_info[1] = np.array([coordinate1[0, 0], coordinate1[0, 1], r1, 1])
        cam.cam_info[3] = np.array([coordinate2[0, 0], coordinate2[0, 1], r2, 1])
        cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, point_e, vector_e_t, 1, 181, 100)
        # print(np.max(torque4), np.min(torque4))
        print(x0, coordinate1, coordinate2, np.min(torque4 + torque2 + torque3), np.max(torque4 + torque2 + torque3), np.ptp(stroke4))
        return np.min(torque4 + torque2 + torque3)**2 + np.max(torque4 + torque2 + torque3)**2

    cam = Cam()
    cam.cam_info = np.array([
        [40, 0, 30, -1],
        [-0.20096282, 8.05497287, 11, 1],
        [-0.12867570, 8.99908010, 10, 1],
        [-0.95139654, 7.31226925, 11, 1],
        [-39.39231012, -6.94592711, 30, -1],
        [0, 0, 10, 1]])

    # x0 = [11.1, 12.7]
    # result = minimize(fun=func1, x0=x0, bounds=[[10.7, 23], [12, 18]])
    # print(result)
    func1([11.1, 12.7])

    cam.cam_info[1] = np.array([-0.33014058, 7.9176866, 11.1, 1])
    cam.cam_info[3] = np.array([1.10327211, 6.59651801, 12.7, 1])
    cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, point_e, vector_e_t, 1, 181, 100)
    return torque4, stroke4


def cam1_version2():
    def func1(x0):
        r1 = x0[0]
        r2 = x0[1]
        # print(x0)
        coordinate1 = solveCoordinate(cam, 1, 0, 2, r1, 1)
        coordinate2 = solveCoordinate(cam, 2, 1, 3, r2, 1)
        cam.cam_info[1] = np.array([coordinate1[0, 0], coordinate1[0, 1], r1, 1])
        cam.cam_info[2] = np.array([coordinate2[0, 0], coordinate2[0, 1], r2, 1])
        cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, point_e, vector_e_t, 1, 181, 100)
        # print(np.max(torque4), np.min(torque4))
        print(x0, coordinate1, coordinate2, np.min(torque4 + torque2 + torque3), np.max(torque4 + torque2 + torque3), np.ptp(stroke4))
        return np.min(torque4 + torque2 + torque3)**2 + np.max(torque4 + torque2 + torque3)**2  # + np.abs(9 - np.ptp(stroke4)) * 1e6

    cam = Cam()
    cam.cam_info = np.array([
        [40, 0, 30, -1],
        [0.02443393, 8.16052190, 10.8, 1],
        [0.86348762, 6.68201288, 12.5, 1],
        [-39.39231012, -6.94592711, 30, -1],
        [0, 0, 10, 1]])

    # x0 = [10.8, 12.5]
    # result = minimize(fun=func1, x0=x0, bounds=[[10, 11], [12, 18]])
    # print(result)
    # func1([10.8, 12.5])

    cam.cam_info[1] = np.array([0.02443393, 8.16052190, 10.8, 1])
    cam.cam_info[2] = np.array([0.86348762, 6.68201288, 12.5, 1])
    cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, point_e, vector_e_t, 1, 181, 100)
    return torque4, stroke4


def cam2_version1():
    def func1(x0):
        r1 = x0[0]
        r2 = x0[1]
        # print(x0)
        coordinate1 = solveCoordinate(cam, 1, 0, 2, r1, 1)
        coordinate2 = solveCoordinate(cam, 2, 1, 3, r2, 1)
        cam.cam_info[1] = np.array([coordinate1[0, 0], coordinate1[0, 1], r1, 1])
        cam.cam_info[2] = np.array([coordinate2[0, 0], coordinate2[0, 1], r2, 1])
        cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, point_e, vector_e_t, 1, 181, 100)
        # print(np.max(torque4), np.min(torque4))
        print(x0, coordinate1, coordinate2, np.min(torque4 + torque2 + torque3), np.max(torque4 + torque2 + torque3), np.ptp(stroke4))
        return np.min(torque4 + torque2 + torque3)**2 + np.max(torque4 + torque2 + torque3)**2  # + np.abs(9 - np.ptp(stroke4)) * 1e6

    cam = Cam()
    cam.cam_info = np.array([
        [8.8749835, 19.03246353, 11, -1],
        [-4.36026430, 0.22214147, 12, 1],
        [-5.91364601, 7.04760888, 5, 1],
        [0, 0, 14.2, 1],
        [-5.91364601, -7.04760888, 5, 1],
        [-4.66767263, -2.20534181, 10, 1],
        [9.57656401, -26.31139338, 18, -1],
        [0, 0, 10, 1]])

    # x0 = [10.8, 12.5]
    # result = minimize(fun=func1, x0=x0, bounds=[[10, 11], [12, 18]])
    # print(result)
    # func1([10, 5])

    # cam.cam_info[1] = np.array([-4.36026430, 0.22214147, 12, 1])
    # cam.cam_info[2] = np.array([-5.91364601, 7.04760888, 5, 1])
    cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, np.array([18.4, 11]), 21, 1, 80, 80)
    return torque4, stroke4


def cam3_version1():
    def func1(x0):
        r1 = x0[0]
        r2 = x0[1]
        # print(x0)
        coordinate1 = solveCoordinate(cam, 0, -1, 1, r1, 1)
        coordinate2 = solveCoordinate(cam, 4, 3, 5, r2, 1)
        cam.cam_info[0] = np.array([coordinate1[0, 0], coordinate1[0, 1], r1, 1])
        cam.cam_info[4] = np.array([coordinate2[0, 0], coordinate2[0, 1], r2, 1])
        cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, np.array([-20.5, 21]), 23, -1, 60, 20)
        print(x0, coordinate1, coordinate2, np.max(torque4), np.min(torque4))
        return np.max(torque4)**2 + np.min(torque4)**2

    cam = Cam()
    cam.cam_info = np.array([
        [-0.77415614, 7.54480370, 14, 1],
        [6.11604817, 8.77974685, 7, 1],
        [0, 0, 17.7, 1],
        [6.03677317, 11.17351197, 5, 1],
        [12.71540877, -7.34124467, 24.68248935, 1],
        [0, 0, 10, 1],
        [28.19077862, -10.2606043, 20, -1]])

    # x0 = [10.4, 14]
    # result = minimize(fun=func1, x0=x0, bounds=[[None, None], [14, 30]])
    # print(result)
    # func1([10.4, 22])

    cam.cam_info[0] = np.array([3.19437391, 7.04083354, 10.4, 1])
    cam.cam_info[3] = np.array([6.25288070, 11.62503690, 4.5, 1])
    cam.cam_info[4] = np.array([11.80736605, -6.54492987, 23.5, 1])
    cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, np.array([-20.5, 21]), 23, -1, 60, 20)
    return torque4, stroke4


def cam3_version2():
    def func1(x0):
        r1 = x0[0]
        # print(x0)
        coordinate1 = solveCoordinate(cam, 2, 1, 3, r1, -1)
        cam.cam_info[2] = np.array([coordinate1[0, 0], coordinate1[0, 1], r1, 1])
        cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, np.array([-20.5, 21]), 23, -1, 60, 20)
        print(x0, coordinate1, np.max(torque4), np.min(torque4))
        return np.max(torque4)**2 + np.min(torque4)**2

    cam = Cam()
    cam.cam_info = np.array([
        [4.55623183, 7.05696475, 9.3, 1],
        [0, 0, 17.7, 1],
        [6.03677317, 11.17351197, 5, 1],
        [13.85640646, -8, 26, 1],
        [0, 0, 10, 1],
        [28.19077862, -10.2606043, 20, -1]])

    # x0 = [7.3]
    # result = minimize(fun=func1, x0=x0, bounds=[[None, None], [14, 30]])
    # print(result)
    # func1([6])

    cam.cam_info[2] = np.array([6.25288070, 11.62503690, 4.5, 1])
    cam.cam_info[3] = np.array([11.80736605, -6.54492987, 23.5, 1])
    cam_1, intersection_point4, torque4, stroke4 = solveCam(cam, np.array([-20.5, 21]), 23, -1, 60, 20)
    return torque4, stroke4


# torque4, stroke4 = cam1_version1()
torque4, stroke4 = cam1_version2()
torque5, stroke5 = cam2_version1()
# torque6, stroke6 = cam3_version1()
torque6, stroke6 = cam3_version2()
torque_total1 = torque4 + torque5 + torque6


def plot2():
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(stroke1 * 100 - 1000, 'r--', linewidth=1)
    ax1.plot(stroke4 * 100 - 1000, 'g', linewidth=1)
    ax1.plot(torque1, 'r--', linewidth=1)
    ax1.plot(torque4, 'g', linewidth=1)
    ax1.set_title("抬压脚凸轮")
    ax1.text(380, 1500, '行程')
    ax1.text(380, 0, '力矩')
    ax1.grid()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(stroke2 * 100 - 1000, 'r--', linewidth=1)
    ax2.plot(stroke5 * 100 - 1000, 'g', linewidth=1)
    ax2.plot(torque2, 'r--', linewidth=1)
    ax2.plot(torque5, 'g', linewidth=1)
    ax2.set_title("针距凸轮")
    ax2.grid()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(stroke3 * 100 - 1000, 'r--', linewidth=1)
    ax3.plot(stroke6 * 100 - 1000, 'g', linewidth=1)
    ax3.plot(torque3, 'r--', linewidth=1)
    ax3.plot(torque6, 'g', linewidth=1)
    ax3.set_title("剪线凸轮")
    ax3.grid()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(torque_total, 'r--', linewidth=1)
    ax4.plot(torque_total1, 'g', linewidth=1)
    ax4.set_title("力矩总和")
    ax4.grid()

    plt.show()


plot2()
