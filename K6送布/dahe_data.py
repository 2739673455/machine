import sys
sys.path.append(sys.path[0] + '\..')
import numpy as np
from function import *
from machine import Machine


class Dahe(Machine):
    def __init__(self):
        self.du = 180 / np.pi
        self.hd = np.pi / 180
        self.n = np.linspace(0, 359 * self.hd, 360).reshape(-1, 1)
        self.tph = -174.519

        # 送布
        self.p4_a = np.array([0, 0, 0])
        self.p4_d2 = np.array([0, 10.00, -49.80])
        self.p4_f = np.array([0, 35.90, -14.40])
        self.p4_w2 = np.array([0, 41.07927576, -6.44819288])
        self.p4_t = np.array([0, 54.00, -10.50])
        self.p4_j = np.array([0, 53.00, 8.00])
        self.p4_y = np.array([0, 58.00, -53.10])

        self.l4_ab3 = 3.8
        self.l4_b3c = 51
        self.l4_cd1 = 20
        self.l4_d1d2 = 20

        self.l4_ce = 40
        self.l4_ef = 15
        self.l4_fg = 16
        self.l4_gh = 14.8

        self.l4_fg2 = 30
        self.l4_g2v = 30

        self.l4_yx = 36
        self.l4_xw = 65.5
        self.l4_wv = 43.3
        self.l4_vu = 30
        self.l4_ut = 20
        self.l4_ts = 15  # 23,15
        self.l4_sh2 = 24.5

        self.l4_ab2 = 2.5
        self.l4_b2i = 52.7
        self.l4_ij = 14
        self.l4_jk = 7
        self.l4_kk2 = 10.4

        self.l4_ab1 = 1

        self.theta4_0d2d3 = -15.3 * self.hd  # 针距 小6.20110532  ~ 大-20.34000173
        self.theta4_d3d2d1 = 20 * self.hd
        self.theta4_efg = 152 * self.hd
        self.theta4_efg2 = 47 * self.hd

        self.theta4_0yz = -9 * self.hd  # 差动 小-9.62300248  ~ 大15.52972141
        self.theta4_zyx = 170 * self.hd
        self.theta4_0w2w = 60 * self.hd
        self.theta4_uts = 157.3756472 * self.hd  # 148.5*hd,157.3756472*hd

        self.theta4_ijk = 51.6 * self.hd
        self.theta4_b1ab2 = -64.9 * self.hd
        self.theta4_b1ab3 = -65.2 * self.hd

    def FabricFeedSolve(self):
        self.theta4_0ab1 = self.n
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

        self.p4_x = p_rl(self.p4_y, self.l4_yx, self.theta4_0yz + self.theta4_zyx, 0)
        self.p4_w, _ = rrp(self.p4_x[[1, 2]], self.p4_w2[[1, 2]], self.l4_xw, 0, self.theta4_0w2w, 0, 1)
        self.p4_w = np.insert(self.p4_w[0], 0, 0)

        self.p4_v, self.theta4_0vg2 = rrr(self.p4_g2[:, 1:3], self.p4_w[[1, 2]], self.l4_g2v, self.l4_wv, -1)
        self.p4_v = np.insert(self.p4_v, 0, 0, axis=1)
        self.p4_u, self.theta4_0tu = rrr(self.p4_v[:, 1:3], self.p4_t[[1, 2]], self.l4_vu, self.l4_ut, 1)
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

        self.p4_h, _ = rrp(self.p4_g[:, 1:3], self.p4_b1[:, 1:3], self.l4_gh, 0.4, self.theta4_0b1k2, -90 * self.hd, -1)
        self.p4_h2, _ = rrp(self.p4_s[:, 1:3], self.p4_b1[:, 1:3], self.l4_sh2, 11.6, self.theta4_0b1k2, -90 * self.hd, -1)
        self.p4_h = np.insert(self.p4_h, 0, 0, axis=1)
        self.p4_h2 = np.insert(self.p4_h2, 0, 0, axis=1)

        self.p4_l1 = p_rp(np.array([0, -15.1, 39.25]), self.theta4_0b1k2, 0) + self.p4_h
        self.p4_l2 = p_rp(np.array([0, 4.7, 39.25]), self.theta4_0b1k2, 0) + self.p4_h
        self.p4_r1 = p_rp(np.array([0, 2.5, 28.05]), self.theta4_0b1k2, 0) + self.p4_h2
        self.p4_r2 = p_rp(np.array([0, 10.75, 28.05]), self.theta4_0b1k2, 0) + self.p4_h2

    def solveTrajectory(self, theta1, theta2):
        self.theta4_0d2d3 = theta1 * self.hd
        self.theta4_0yz = theta2 * self.hd
        self.FabricFeedSolve()


if __name__ == '__main__':
    dahe = Dahe()
    # dahe.n = np.array([[23.44889345 * hd]])
    dahe.solveTrajectory(-15.3, -9)
    print(dahe.p4_l1)
