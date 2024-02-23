import sys
sys.path.append('./..')
import numpy as np
from function import *
from machine import Machine


class JackSource(Machine):
    def __init__(self):
        super(JackSource, self).__init__()
        self.theta4_b1ab2 = -66.5 * self.hd
        self.theta4_b1ab3 = -62.5 * self.hd  # -57.5

    def solveTrajectory(self, theta1, theta2):
        self.theta4_0d2d3 = theta1 * self.hd
        self.theta4_0xw = theta2 * self.hd
        self.FabricFeedSolve()


class JackV1(Machine):
    def __init__(self):
        super(JackV1, self).__init__()
        self.theta4_b1ab2 = -66.5 * self.hd
        self.theta4_b1ab3 = -62.5 * self.hd  # -57.5
        self.l4_xw = 48
        self.l4_wv = 28
        self.l4_ts = 15
        self.theta4_uts = -134.17943223 * self.hd  # -147.154682283 * hd
        self.theta4_efg2 = -31 * self.hd  # -33.8,-31

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

        self.p4_w = p_rl(self.p4_x, self.l4_xw, self.theta4_0xw, 0)
        self.p4_v, self.theta4_0wv = rrr(self.p4_g2[:, 1:3], self.p4_w[[1, 2]], self.l4_g2v, self.l4_wv, 1)  # version 1
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

        self.p4_l1 = p_rp(np.array([0, -12.8, 39.75]), self.theta4_0b1k2, 0) + self.p4_h
        self.p4_l2 = p_rp(np.array([0, 3.7, 39.75]), self.theta4_0b1k2, 0) + self.p4_h
        self.p4_r1 = p_rp(np.array([0, 5.25, 28.25]), self.theta4_0b1k2, 0) + self.p4_h2
        self.p4_r2 = p_rp(np.array([0, 14.25, 28.25]), self.theta4_0b1k2, 0) + self.p4_h2

    def solveTrajectory(self, theta1, theta2):
        self.theta4_0d2d3 = theta1 * self.hd
        self.theta4_0xw = theta2 * self.hd
        self.FabricFeedSolve()
