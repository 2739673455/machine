import sys
sys.path.append('./..')
import numpy as np
from function import *
from scipy.optimize import root

np.set_printoptions(linewidth=200)
du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)


def k6_solve(p, l, theta, n):
    theta['4_0ab1'] = n
    theta['4_0ab2'] = theta['4_0ab1'] + theta['4_b1ab2']
    theta['4_0ab3'] = theta['4_0ab1'] + theta['4_b1ab3']
    p['4_d1'] = p_rl(p['4_d2'], l['4_d1d2'], theta['4_0d2d3'] + theta['4_d3d2d1'], 0)

    p['4_b3'] = p_rl(p['4_a'], l['4_ab3'], theta['4_0ab3'], 0)
    p['4_c'], theta['4_0d1c'] = rrr(p['4_b3'][:, 1:3], p['4_d1'][[1, 2]], l['4_b3c'], l['4_cd1'], 1)
    p['4_c'] = np.insert(p['4_c'], 0, 0, axis=1)
    p['4_e'], theta['4_0fe'] = rrr(p['4_c'][:, 1:3], p['4_f'][[1, 2]], l['4_ce'], l['4_ef'], 1)
    p['4_e'] = np.insert(p['4_e'], 0, 0, axis=1)
    theta['4_0fg'] = theta['4_0fe'] + theta['4_efg']
    p['4_g'] = p_rl(p['4_f'], l['4_fg'], theta['4_0fg'], 0)

    theta['4_0fg2'] = theta['4_0fe'] + theta['4_efg2']
    p['4_g2'] = p_rl(p['4_f'], l['4_fg2'], theta['4_0fg2'], 0)
    # p['4_x1'] = p_rl(p['4_y'], l['4_yx1'], theta['4_0yx1'], 0)
    # p['4_x2'], theta['4_0x3x2'] = rrr(p['4_x1'][[1, 2]], p['4_x3'][[1, 2]], l['4_x1x2'], l['4_x2x3'], -1)
    # p['4_w'] = p_rl(p['4_x3'], l['4_x3w'], theta['4_0x3x2'][0][0]+theta['4_x2x3w'], 0)
    p['4_w'] = p_rl(p['4_x3'], l['4_x3w'], theta['4_0x3w'], 0)

    p['4_v'], theta['4_0wv'] = rrr(p['4_g2'][:, 1:3], p['4_w'][[1, 2]], l['4_g2v'], l['4_wv'], -1)
    p['4_v'] = np.insert(p['4_v'], 0, 0, axis=1)
    p['4_u'], theta['4_0tu'] = rrr(p['4_v'][:, 1:3], p['4_t'][[1, 2]], l['4_vu'], l['4_ut'], 1)

    theta['4_0ts'] = theta['4_0tu'] + theta['4_uts']
    p['4_s'] = p_rl(p['4_t'], l['4_ts'], theta['4_0ts'], 0)

    p['4_b2'] = p_rl(p['4_a'], l['4_ab2'], theta['4_0ab2'], 0)
    p['4_i'], theta['4_0ji'] = rrr(p['4_b2'][:, 1:3], p['4_j'][[1, 2]], l['4_b2i'], l['4_ij'], -1)
    theta['4_0jk'] = theta['4_0ji'] + theta['4_ijk']
    p['4_k'] = p_rl(p['4_j'], l['4_jk'], theta['4_0jk'], 0)

    p['4_b1'] = p_rl(p['4_a'], l['4_ab1'], theta['4_0ab1'], 0)
    theta['4_0b1k'], l['4_b1k'] = theta_l(p['4_k'][:, 1:3] - p['4_b1'][:, 1:3])
    theta['4_kb1k2'] = -np.arcsin(l['4_kk2'] / l['4_b1k'])
    theta['4_0b1k2'] = theta['4_0b1k'] + theta['4_kb1k2']

    p['4_h'], _ = rrp(p['4_g'][:, 1:3], p['4_b1'][:, 1:3], l['4_gh'], 1, theta['4_0b1k2'], 90 * hd, -1)
    p['4_h2'], _ = rrp(p['4_s'][:, 1:3], p['4_b1'][:, 1:3], l['4_sh2'], l['4_h2'], theta['4_0b1k2'], -90 * hd, -1)
    p['4_h'] = np.insert(p['4_h'], 0, 0, axis=1)
    p['4_h2'] = np.insert(p['4_h2'], 0, 0, axis=1)

    p['4_l1'] = p_rp(np.array([0, -12.8, 39.75]), theta['4_0b1k2'], 0) + p['4_h']
    p['4_l2'] = p_rp(np.array([0, 3.7, 39.75]), theta['4_0b1k2'], 0) + p['4_h']
    p['4_r1'] = p_rp(np.array([0, 3.75, 38.75 - l['4_h2']]), theta['4_0b1k2'], 0) + p['4_h2']
    p['4_r2'] = p_rp(np.array([0, 12.75, 38.75 - l['4_h2']]), theta['4_0b1k2'], 0) + p['4_h2']


def k6_solve2(p, l, theta, n):  # 主轴角度固定，差动角度为一个范围
    theta['4_0ab1'] = n
    theta['4_0ab2'] = theta['4_0ab1'] + theta['4_b1ab2']
    theta['4_0ab3'] = theta['4_0ab1'] + theta['4_b1ab3']
    p['4_d1'] = p_rl(p['4_d2'], l['4_d1d2'], theta['4_0d2d3'] + theta['4_d3d2d1'], 0)

    p['4_b3'] = p_rl(p['4_a'], l['4_ab3'], theta['4_0ab3'], 0)
    p['4_c'], theta['4_0d1c'] = rrr(p['4_b3'][:, 1:3], p['4_d1'][[1, 2]], l['4_b3c'], l['4_cd1'], 1)
    p['4_c'] = np.insert(p['4_c'], 0, 0, axis=1)
    p['4_e'], theta['4_0fe'] = rrr(p['4_c'][:, 1:3], p['4_f'][[1, 2]], l['4_ce'], l['4_ef'], 1)
    p['4_e'] = np.insert(p['4_e'], 0, 0, axis=1)
    theta['4_0fg'] = theta['4_0fe'] + theta['4_efg']
    p['4_g'] = p_rl(p['4_f'], l['4_fg'], theta['4_0fg'], 0)

    theta['4_0fg2'] = theta['4_0fe'] + theta['4_efg2']
    p['4_g2'] = p_rl(p['4_f'], l['4_fg2'], theta['4_0fg2'], 0)
    # p['4_x1'] = p_rl(p['4_y'], l['4_yx1'], theta['4_0yx1'], 0)
    # p['4_x2'], theta['4_0x3x2'] = rrr(p['4_x1'][[1, 2]], p['4_x3'][[1, 2]], l['4_x1x2'], l['4_x2x3'], -1)
    # p['4_w'] = p_rl(p['4_x3'], l['4_x3w'], theta['4_0x3x2'][0][0]+theta['4_x2x3w'], 0)
    p['4_w'] = p_rl(p['4_x3'], l['4_x3w'], theta['4_0x3w'], 0)

    p['4_v'], theta['4_0g2v'] = rrr(p['4_w'][:, 1:3], p['4_g2'][:, 1:3], l['4_wv'], l['4_g2v'], 1)
    p['4_v'] = np.insert(p['4_v'], 0, 0, axis=1)
    p['4_u'], theta['4_0tu'] = rrr(p['4_v'][:, 1:3], p['4_t'][[1, 2]], l['4_vu'], l['4_ut'], 1)

    theta['4_0ts'] = theta['4_0tu'] + theta['4_uts']
    p['4_s'] = p_rl(p['4_t'], l['4_ts'], theta['4_0ts'], 0)

    p['4_b2'] = p_rl(p['4_a'], l['4_ab2'], theta['4_0ab2'], 0)
    p['4_i'], theta['4_0ji'] = rrr(p['4_b2'][:, 1:3], p['4_j'][[1, 2]], l['4_b2i'], l['4_ij'], -1)
    theta['4_0jk'] = theta['4_0ji'] + theta['4_ijk']
    p['4_k'] = p_rl(p['4_j'], l['4_jk'], theta['4_0jk'], 0)

    p['4_b1'] = p_rl(p['4_a'], l['4_ab1'], theta['4_0ab1'], 0)
    theta['4_0b1k'], l['4_b1k'] = theta_l(p['4_k'][:, 1:3] - p['4_b1'][:, 1:3])
    theta['4_kb1k2'] = -np.arcsin(l['4_kk2'] / l['4_b1k'])
    theta['4_0b1k2'] = theta['4_0b1k'] + theta['4_kb1k2']

    p['4_h'], _ = rrp(p['4_g'][:, 1:3], p['4_b1'][:, 1:3], l['4_gh'], 1, theta['4_0b1k2'], 90 * hd, -1)
    p['4_h2'], _ = rrp(p['4_s'][:, 1:3], p['4_b1'][:, 1:3], l['4_sh2'], l['4_h2'], theta['4_0b1k2'], -90 * hd, -1)
    p['4_h'] = np.insert(p['4_h'], 0, 0, axis=1)
    p['4_h2'] = np.insert(p['4_h2'], 0, 0, axis=1)

    p['4_l1'] = p_rp(np.array([0, -12.8, 39.75]), theta['4_0b1k2'], 0) + p['4_h']
    p['4_l2'] = p_rp(np.array([0, 3.7, 39.75]), theta['4_0b1k2'], 0) + p['4_h']
    p['4_r1'] = p_rp(np.array([0, 3.75, 38.75 - l['4_h2']]), theta['4_0b1k2'], 0) + p['4_h2']
    p['4_r2'] = p_rp(np.array([0, 12.75, 38.75 - l['4_h2']]), theta['4_0b1k2'], 0) + p['4_h2']


if __name__ == '__main__':
    from dahe_data import Dahe
    from k6_data2 import *
    from scipy.optimize import minimize

    dahe = Dahe()
    dahe.solveTrajectory(-6.933, 15.5)  # 针距 小6.2 ~ 大-20.3,差动 小-9.6 ~ 大15.5

    jack = dict()
    k6_data2(p, l, theta)
    theta['4_0d2d3'] = 1 * hd  # 针距 小12 ~ 大-13 3mm:-2.75
    theta['4_0x3w'] = -12.7 * hd  # 差动
    l['4_ts'] = 18.5
    l['4_h2'] = 15.5  # 10.5/15.5
    left_bound = 30.4
    right_bound = 45.9

    # version 1
    # l['4_sh2'], l['4_x3w'], l['4_vu'], theta['4_uts'], theta['4_0x3w'] = [26, 36, 16, 188 * hd, -19.50950128 * hd]
    l['4_sh2'], l['4_x3w'], l['4_vu'], theta['4_uts'], theta['4_0x3w'] = [26, 34, 16, 188 * hd, -19.50950128 * hd]
    theta_list = [[-2.75 * hd, -20.45 * hd, 3], [-10 * hd, -15.38 * hd, 4], [-13 * hd, -13.38 * hd, 4.4]]  # 针距角度、差动角度、针距
    # version 1

    def travel1(angle, args):  # 计算给定差动角度下，差动牙行程与6mm的差距，args:差动牙行程
        theta['4_0x3w'] = angle * hd
        k6_solve(p, l, theta, n)
        # print(angle)
        return (np.ptp(p['4_r1'][:, 1]) - args)**2

    # 计算满足差动行程的差动角度
    # theta['4_0d2d3'], _, distance1 = theta_list[2]
    # angle1 = -14
    # result = minimize(fun=travel1, x0=angle1, args=6)
    # angle1 = result.x[0]
    # print("差动6mm对应角度:", angle1, result.fun)
    # angle2 = -10
    # result = minimize(fun=travel1, x0=angle2, args=distance1 * 0.5)
    # angle2 = result.x[0]
    # print("差动为0.5针距对应角度:", angle2, result.fun)

    # for theta['4_0d2d3'], theta['4_0x3w'], _ in theta_list:  # 差动6mm时，整周转动，差动牙到针板左右最小距离
    #     k6_solve(p, l, theta, n)
    #     print("针距", np.ptp(p['4_l1'][:, 1]), "| 差动", np.ptp(p['4_r1'][:, 1]))
    #     print("距针板左最小: ", np.min(p['4_r1'][:, 1]) + 4.5 - left_bound)
    #     print("距针板右最小: ", right_bound - (np.max(p['4_r1'][:, 1]) + 10.3), end="\n\n")

    def distance1(n, angle1, angle2):
        # 差动调节范围
        theta['4_0x3w'] = angle1 * hd
        k6_solve(p, l, theta, n)
        print("差动:", np.ptp(p['4_r1'][:, 1]), "角度:", angle1)
        theta['4_0x3w'] = angle2 * hd
        k6_solve(p, l, theta, n)
        print("差动:", np.ptp(p['4_r1'][:, 1]), "角度:", angle2)
        # 差动角度范围内差动牙平移距离
        n = np.array([[178 * hd]])
        theta['4_0x3w'] = np.linspace(angle1 * hd, angle2 * hd, int((angle2 - angle1) * 10)).reshape(-1, 1)  # 差动
        k6_solve2(p, l, theta, n)
        print("差动牙平移距离:", p['4_r1'][:, 1].ptp())
        # 差动角度范围内差动牙与针板左侧最小距离
        n = np.array([[179 * hd]])
        theta['4_0x3w'] = np.linspace(angle1 * hd, angle2 * hd, int((angle2 - angle1) * 10)).reshape(-1, 1)  # 差动
        k6_solve2(p, l, theta, n)
        print("距针板左侧:", np.min(p['4_r1'][:, 1]) + 4.5 - left_bound)
        # 差动角度范围内差动牙与针板右侧最小距离
        n = np.array([[348 * hd]])
        theta['4_0x3w'] = np.linspace(angle1 * hd, angle2 * hd, int((angle2 - angle1) * 10)).reshape(-1, 1)  # 差动
        k6_solve2(p, l, theta, n)
        print("距针板右侧:", right_bound - (np.max(p['4_r1'][:, 1]) + 10.3))

    angle1 = -30
    angle2 = 1
    distance1(n=n, angle1=angle1, angle2=angle2)

    theta['4_0x3w'] = -6 * hd
    k6_solve(p, l, theta, n)
    dahe.p4_l1 = dahe.p4_l1 - dahe.p4_l1[0] + p['4_l1'][0]
    dahe.p4_r1 = dahe.p4_r1 - dahe.p4_r1[0] + p['4_r1'][0]
    jack['l1'] = p['4_l1']
    jack['r1'] = p['4_r1']
    print("杰克|针距", np.ptp(jack['l1'][:, 1]), " 差动", np.ptp(jack['r1'][:, 1]))
    print("大和|针距", np.ptp(dahe.p4_l1[:, 1]), " 差动", np.ptp(dahe.p4_r1[:, 1]))
    print(p['4_l1'][-1])
    print(p['4_r1'][-1])

    # trace_list = []
    # for var1 in np.linspace(angle1, angle2, 20):
    #     theta['4_0x3w'] = var1 * hd
    #     k6_solve(p, l, theta, n)
    #     trace_list.append(p['4_r1'])
    # k6_plot(dahe, trace_list, n)
