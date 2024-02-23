import sys
sys.path.append('./..')
import numpy as np
import matplotlib.pyplot as plt
from function import *
from dahe_data import *
from machine import Machine
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import axes3d

np.set_printoptions(linewidth=200)
du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)


class Jack1(Machine):
    def FabricFeedSolve(self):
        self.theta4_0ab1 = self.theta1_0ab + self.theta4_14 + self.n
        self.theta4_0ab2 = self.theta4_0ab1 + self.theta4_b1ab2
        self.theta4_0ab3 = self.theta4_0ab1 + self.theta4_b1ab3
        self.p4_d1 = p_rl(self.p4_d2, self.l4_d1d2, self.theta4_0d2d3 + self.theta4_d3d2d1, 0)

        self.p4_b3 = p_rl(self.p4_a, self.l4_ab3, self.theta4_0ab3, 0)
        self.p4_c, self.theta4_0d1c = rrr(self.p4_b3[:, 1:3], self.p4_d1[[1, 2]], self.l4_b3c, self.l4_cd1, 1)
        self.p4_c = np.insert(self.p4_c, 0, 0, axis=1)
        self.p4_e, self.theta4_0fe = rrr(self.p4_c[:, 1:3], self.p4_f[[1, 2]], self.l4_ce, self.l4_ef, 1)
        self.p4_e = np.insert(self.p4_e, 0, 0, axis=1)
        self.theta4_0fg = self.theta4_0fe + self.theta4_efg
        self.p4_g = p_rl(self.p4_f, self.l4_fg, self.theta4_0fg, 0)

        self.theta4_0fg2 = self.theta4_0fe + self.theta4_efg2
        self.p4_g2 = p_rl(self.p4_f, self.l4_fg2, self.theta4_0fg2, 0)
        self.p4_w = p_rl(self.p4_x, self.l4_xw, self.theta4_0xw, 0)

        self.p4_v, self.theta4_0wv = rrr(self.p4_g2[:, 1:3], self.p4_w[[1, 2]], self.l4_g2v, self.l4_wv, 1)
        self.p4_v = np.insert(self.p4_v, 0, 0, axis=1)
        self.p4_u, self.theta4_0tu = rrr(self.p4_v[:, 1:3], self.p4_t[[1, 2]], self.l4_vu, self.l4_ut, -1)
        self.theta4_0ts = self.theta4_0tu + self.theta4_uts
        self.p4_s = p_rl(self.p4_t, self.l4_ts, self.theta4_0ts, 0)

        self.p4_b2 = p_rl(self.p4_a, self.l4_ab2, self.theta4_0ab2, 0)
        self.p4_i, self.theta4_0ji = rrr(self.p4_b2[:, 1:3], self.p4_j[[1, 2]], self.l4_b2i, self.l4_ij, -1)
        self.theta4_0jk = self.theta4_0ji + self.theta4_ijk
        self.p4_k = p_rl(self.p4_j, self.l4_jk, self.theta4_0jk, 0)

        self.p4_b1 = p_rl(self.p4_a, self.l4_ab1, self.theta4_0ab1, 0)
        self.theta4_0b1k, self.l4_b1k = theta_l(self.p4_k[:, 1:3] - self.p4_b1[:, 1:3])
        self.theta4_kb1k2 = -np.arcsin(self.l4_kk2 / self.l4_b1k)
        self.theta4_0b1k2 = self.theta4_0b1k + self.theta4_kb1k2

        self.p4_h, _ = rrp(self.p4_g[:, 1:3], self.p4_b1[:, 1:3], self.l4_gh, 1, self.theta4_0b1k2, 90 * self.hd, -1)
        self.p4_h2, _ = rrp(self.p4_s[:, 1:3], self.p4_b1[:, 1:3], self.l4_sh2, 10.5, self.theta4_0b1k2, -90 * self.hd, -1)
        self.p4_h = np.insert(self.p4_h, 0, 0, axis=1)
        self.p4_h2 = np.insert(self.p4_h2, 0, 0, axis=1)

        self.p4_l1 = p_rp(self.p4_l1_, self.theta4_0b1k2, 0) + self.p4_h
        self.p4_l2 = p_rp(self.p4_l2_, self.theta4_0b1k2, 0) + self.p4_h
        self.p4_r1 = p_rp(self.p4_r1_, self.theta4_0b1k2, 0) + self.p4_h2
        self.p4_r2 = p_rp(self.p4_r2_, self.theta4_0b1k2, 0) + self.p4_h2


def dahe_trajectory(theta1, theta2):  # 大和轨迹
    dahe_data(p, l, theta)
    theta['4_0d2d3'] = theta1 * hd  # 针距 小6.20110532 ~ 大-20.34000173
    theta['4_0yz'] = theta2 * hd  # 差动 小-9.62300248 ~ 大15.52972141
    dahe_solve(p, l, theta, n)
    return p['4_l1'], p['4_r1']


def shifting():
    jack0.p4_l1 = jack0.p4_l1 - jack0.p4_l1[0] + dahe['l1'][0]
    jack0.p4_r1 = jack0.p4_r1 - jack0.p4_r1[0] + dahe['r1'][0]
    jack1.p4_l1 = jack1.p4_l1 - jack1.p4_l1[0] + dahe['l1'][0]
    jack1.p4_r1 = jack1.p4_r1 - jack1.p4_r1[0] + dahe['r1'][0]


def plot1(dahe, jack, n):
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_aspect(1)

    dahe_line1 = ax1.plot(dahe['r1'][:, 1], dahe['r1'][:, 2], 'g', lw=1, label="大和")
    jack_line1 = ax1.plot(jack['r1'][:, 1], jack['r1'][:, 2], 'b', lw=1, label="杰克")
    ax1.legend()

    plt.show()


dahe = dict()
dahe['l1'], dahe['r1'] = dahe_trajectory(-17.3, -6.5)  # 针距 小6.2 ~ 大-20.3,差动 小-9.6 ~ 大15.5

# n = np.array([[62.25271918 * hd]])

# jack0 = Machine()
# jack0.n = n
# jack0.theta4_0d2d3 = -13 * hd  # 针距 小12 ~ 大-13
# jack0.theta4_0xw = -2 * hd  # 差动 小-25 ~ 大-2
# jack0.FabricFeedSolve()

jack1 = Jack1()
jack1.n = n
jack1.p4_x = np.array([0, 20, -53])
jack1.l4_xw = 51.5
jack1.l4_wv = 28
jack1.l4_ts = 23  # 23,15
jack1.theta4_uts = -143.12684762 * hd  # -143.12684762,-130.15159757
jack1.theta4_efg2 = -30.5 * hd  # -33.8 * hd

jack1.theta4_0d2d3 = -13 * hd  # 针距 小12 ~ 大-13
jack1.theta4_0xw = 68 * hd  # 差动 小48 ~ 大68
jack1.FabricFeedSolve()

# shifting()
# print("大和|针距", np.ptp(dahe['l1'][:, 1]), " 差动", np.ptp(dahe['r1'][:, 1]))
# print("杰克0|针距", np.ptp(jack0.p4_l1[:, 1]), " 差动", np.ptp(jack0.p4_r1[:, 1]))
print("杰克1|针距", np.ptp(jack1.p4_l1[:, 1]), " 差动", np.ptp(jack1.p4_r1[:, 1]))
print("差动比", np.ptp(jack1.p4_r1[:, 1]) / np.ptp(jack1.p4_l1[:, 1]))
# print(jack1.p4_l1, jack1.p4_r1)
# plot1(dahe, jack1, n)
