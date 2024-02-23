import sys
sys.path.append('./..')
import numpy as np
from k6_data import K6
from function import *
from scipy.optimize import minimize

du = 180 / np.pi
hd = np.pi / 180
np.set_printoptions(suppress=True)


def SynchronousAngle(machine):  # 求对弯针同步两个角度
    machine_l1_ac_1 = machine.l1_ab + machine.l1_bc - 9  # 正转高9mm
    machine_l1_ac_2 = machine.l1_ab + machine.l1_bc - 9.8  # 反转高9.8mm
    synchronous_angle1 = np.arccos((machine.l1_ab**2 + machine_l1_ac_1**2 - machine.l1_bc**2) / (2 * machine.l1_ab * machine_l1_ac_1))  # 正转角度
    synchronous_angle2 = np.arccos((machine.l1_ab**2 + machine_l1_ac_2**2 - machine.l1_bc**2) / (2 * machine.l1_ab * machine_l1_ac_2))  # 反转角度
    return np.array([[synchronous_angle1 + np.pi, -synchronous_angle2 + np.pi]]).T


def LooperSynchronous(x, machine):  # 对弯针同步
    machine.theta3_13 = x[0] * hd
    machine.Solve()
    return (machine.p3_k[0][0] - machine.p3_k[1][0])**2


def LooperLimitPosition(machine):  # 弯针极限位置
    limit_angle1 = 83.10412921 - machine.theta3_13 * du - machine.theta1_0ab * du + 360  # 弯针左极限点角度
    limit_angle2 = 265.65255149 - machine.theta3_13 * du - machine.theta1_0ab * du  # 弯针右极限点角度
    machine.n = np.array([[limit_angle2 * hd], [limit_angle1 * hd]])
    machine.Solve()
    print("弯针左极限时角度:", limit_angle1, " 弯针右极限时角度:", limit_angle2)
    print("弯针开档:", -2.8 - machine.p3_k[0, 0])
    print("弯针行程:", np.diff(machine.p3_k[:, 0])[0])


def solve1(machine):
    # 1.弯针尖勾机针左边缘时
    p3_k_x = np.array([[-2.4, 0.4, 3.2]]).T
    machine.n = machine.InvLooperSolve(p3_k_x, 1)
    angle1 = machine.n
    machine.Solve()
    distance1 = [machine.p1_e3_top[0][2] - machine.p3_k[0][2],
                 machine.p1_e2_top[1][2] - machine.p3_k[1][2],
                 machine.p1_e1_top[2][2] - machine.p3_k[2][2]]  # 弯针下边缘到机针孔上边缘距离
    distance2 = [machine.p1_f3[0][2] + machine.l1_ab + machine.l1_bc + machine.l1_cd + machine.l1_df2 - machine.l1_e2e3,
                 machine.p1_f2[1][2] + machine.l1_ab + machine.l1_bc + machine.l1_cd + machine.l1_df2,
                 machine.p1_f1[2][2] + machine.l1_ab + machine.l1_bc + machine.l1_cd + machine.l1_df2 + machine.l1_e1e2]  # 机针回升量

    # 2.弯针尖与针孔上边缘重合
    def function1(x):
        machine.n = np.array([x]).T * hd
        machine.Solve()
        return (machine.p3_k[0][2] - machine.p1_e3_top[0][2])**2 + (machine.p3_k[1][2] - machine.p1_e2_top[1][2])**2 + (machine.p3_k[2][2] - machine.p1_e1_top[2][2])**2

    x0 = [221, 232, 241]
    result = minimize(fun=function1, x0=x0, tol=1e-15)
    angle2 = result.x
    machine.n = np.array([angle2]).T * hd
    machine.Solve()
    distance3 = [-2.8 - machine.p3_k[0][0],
                 0 - machine.p3_k[1][0],
                 2.8 - machine.p3_k[2][0]]  # 弯针尖到机针中心距离

    # 3.勾线时弯针孔左边缘与机针左边缘重合时
    p3_k_x = np.array([[1.19986292, 3.99986292, 6.79986292]]).T
    machine.n = machine.InvLooperSolve(p3_k_x, 1)
    machine.Solve()
    distance4 = [machine.p1_e3_bottom[0][2] - machine.p3_j_bottom[0][2],
                 machine.p1_e2_bottom[1][2] - machine.p3_j_bottom[1][2],
                 machine.p1_e1_bottom[2][2] - machine.p3_j_bottom[2][2]]  # 弯针孔下边缘到机针孔下边缘距离

    # 4.弯针尖脱机针左边缘时
    p3_k_x = np.array([[3.2, 0.4, -2.4]]).T
    machine.n = machine.InvLooperSolve(p3_k_x, -1)
    angle3 = machine.n
    machine.Solve()
    distance5 = [machine.p1_e1_top[0][2] - machine.p3_k[0][2],
                 machine.p1_e2_top[1][2] - machine.p3_k[1][2],
                 machine.p1_e3_top[2][2] - machine.p3_k[2][2]]  # 弯针下边缘到机针孔上边缘距离

    # 5.脱线时弯针孔左边缘与机针左边缘重合时
    p3_k_x = np.array([[6.79986292, 3.99986292, 1.19986292]]).T
    machine.n = machine.InvLooperSolve(p3_k_x, -1)
    machine.Solve()
    distance6 = [machine.p1_e1_bottom[0][2] - machine.p3_j_bottom[0][2],
                 machine.p1_e2_bottom[1][2] - machine.p3_j_bottom[1][2],
                 machine.p1_e3_bottom[2][2] - machine.p3_j_bottom[2][2]]  # 弯针孔下边缘到机针孔下边缘距离

    # 6.机针尖与弯针上方齐平时
    def function2(x):
        machine.n = np.array([x]).T * hd
        machine.Solve()
        return (machine.p3_j_top[0][2] - machine.p1_f1[0][2])**2 + (machine.p3_j_top[1][2] - machine.p1_f2[1][2])**2 + (machine.p3_j_top[2][2] - machine.p1_f3[2][2])**2

    x0 = [82, 91, 101]
    result = minimize(fun=function2, x0=x0, tol=1e-15)
    angle4 = result.x
    machine.n = np.array([angle4]).T * hd
    machine.Solve()
    distance7 = [2.8 - machine.p3_k[0][0],
                 0 - machine.p3_k[1][0],
                 -2.8 - machine.p3_k[2][0]]  # 弯针尖到机针中心距离

    machine_result = np.vstack([angle1.reshape(-1) * du + 360, distance1, distance2, angle2, distance3,
                               distance4, angle3.reshape(-1) * du + 360, distance5, distance6, distance7])
    return machine_result


def OptimizeFunction(x0, args):
    # 弯针
    args[1].p3_k_ = np.array([x0[0], -2.09457126, 15.8])  # 弯针尖
    args[1].theta3_13 = x0[1] * hd  # 同步角度
    args[1].theta3_fhi = x0[2] * hd  # 弯针滑动曲柄与滑动摆杆角度 -103.5
    result = solve1(args[1])
    print(result)
    # print(result - args[0])
    # print(x0)
    return np.sum((result - args[0])**2)
    # return (result[1, 2] - args[0][1, 2])**2 + (result[2, 0] - args[0][2, 0])**2 + (result[4, 1] - args[0][4, 1])**2 + (result[5, 0] - args[0][5, 0])**2 + (result[7, 0] - args[0][7, 0])**2 + (result[9, 0] - args[0][9, 0])**2


def ConstraintsFunction1(x0):
    limit_angle2 = 265.65255149 - x0[1] - 90
    temp = K6()
    temp.n = np.array([[limit_angle2 * hd]])
    temp.l1_ab = 16.7
    temp.l1_cd = 77.419
    temp.p3_k_ = np.array([x0[0], -2.09457126, 15.8])  # 弯针尖
    temp.theta3_13 = x0[1] * hd
    temp.theta3_fhi = x0[2] * hd
    temp.Solve()
    return -2.8 - temp.p3_k[0, 0] - 3


source = K6()
# source_result = solve1(source)
# print(source_result)
### 对弯针同步###
# source.n = SynchronousAngle(source)
# x0 = [-3.380083]
# result = minimize(LooperSynchronous, x0, args=source, tol=1e-15)
# print(result)
# print(result.x)
### 对弯针同步###

dest1 = K6()
dest1.l1_ab = 16.7
dest1.l1_cd = 77.419
dest1.p3_k_ = np.array([164.30152435, -2.09457126, 15.8])  # 弯针尖
dest1.theta3_13 = -3.42369768 * hd
### 对弯针同步###
# dest1.n = SynchronousAngle(dest1)
# x0 = [-3.42369768]
# result = minimize(LooperSynchronous, x0, args=dest1, tol=1e-15)
# print(result)
# print(result.x)
### 对弯针同步###
# x0 = [165.42420266, -3.78981852, -102.5]
# dest1.p3_k_[0] = x0[0]  # 弯针尖
# dest1.theta3_13 = x0[1] * hd
# dest1.theta3_fhi = x0[2] * hd
# dest1.Solve()
# print("同步:", np.diff(dest1.p3_k[:, 0]))  # 同步位置时弯针距离差
# LooperLimitPosition(dest1)  # 弯针极限位置

# x0 = [164.30152435, -3.42369768, -103.5]
# cons = ({"type": "ineq", "fun": ConstraintsFunction1})
# result = minimize(fun=OptimizeFunction, x0=x0, args=[source_result, dest1], constraints=cons, bounds=[(None, None), (None, None), (-104.5, -102.5)])
# print(result)
# print(result.x)
# x0 = [165.32575523, -3.42542066, -102.5]  # 改后
# x0 = [165.42420266, -3.78981852, -102.5]  # 改后
# result = OptimizeFunction(x0, [source_result, dest1])
# print(result)

k5 = K6()
k5.l1_ab = 16.7
k5.l1_cd = 77.919
k5.p3_k_ = np.array([164.50152435, -2.09457126, 15.3])  # 弯针尖
k5.theta3_13 = -3.42369768 * hd
# k5_result = solve1(k5)
# print(k5_result)
### 对弯针同步###
# k5.n = SynchronousAngle(k5)
# x0 = [-3.42369768]
# result = minimize(LooperSynchronous, x0, args=k5, tol=1e-15)
# print(result)
# print(result.x)
### 对弯针同步###
# LooperLimitPosition(k5)  # 弯针极限位置
