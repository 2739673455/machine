import sys
sys.path.append(sys.path[0] + '\..')
import numpy as np
from function import *
from machine import Machine


class K6(Machine):
    def __init__(self):
        self.du = 180 / np.pi
        self.hd = np.pi / 180
        self.n = np.linspace(0, 359 * self.hd, 360).reshape(-1, 1)
        self.l_tph = -174.519  # Throat Plate height 针板高度
        # 机针
        self.p1_a = np.array([0, 0, 0])  # 上轴轴心
        self.p1_g2_ = np.array([0, 4, -17])  # 针杆座穿线孔位置
        self.l1_ab = 15.5  # 针杆曲柄
        self.l1_bc = 53  # 针杆连杆
        self.l1_cd = 78.219  # 针杆连杆到针杆座
        self.l1_dd1 = 2.8  # 相邻针间距
        self.l1_e1e2 = 2.4  # 左中针孔高低差
        self.l1_e2e3 = 2.7  # 中右针孔高低差
        self.l1_de2 = 43.7  # 中针孔到针杆座
        self.l1_df2 = 48.9  # 中针尖到针杆座
        self.theta1_0ab = 90 * self.hd  # 初始角度
        # 弯针
        self.p3_a = np.array([-193.6, -28, -213.319])  # 主轴轴心
        self.p3_d = np.array([-193.6, 1.8, -198.319])  # 弯针滑杆轴心
        self.p3_g = np.array([-167, -31.25, -248.319])  # 弯针滑动曲柄轴心
        self.p3_h = np.array([-167, 1.8, -248.319])  # 弯针滑动摆杆轴心
        self.p3_k_ = np.array([164.30152435, -2.09457126, 16])  # 弯针尖
        self.l3_ab = 1.3  # 弯针摆动偏心
        self.l3_bc = 25  # 弯针摆动连杆
        self.l3_cd = 18.48887640  # 弯针摆动摆杆
        self.l3_ae = 7.9  # 弯针滑动偏心
        self.l3_ef = 35  # 弯针滑动连杆
        self.l3_fg = 26.4  # 弯针滑动曲柄
        self.l3_hi = 53  # 弯针滑动摆杆
        self.theta3_13 = -3.380083 * self.hd  # 机针曲柄与弯针滑动偏心角度
        self.theta3_bae = 130.5 * self.hd  # 弯针滑动偏心与摆动偏心角度
        self.theta3_fhi = -103.5 * self.hd  # 弯针滑动曲柄与滑动摆杆角度
        self.theta3_cdy = -161.5 * self.hd  # 弯针摆动摆杆与弯针架y方向角度

    def Solve(self):
        self.NeedleBarSolve()  # 针杆
        self.LooperSolve()  # 弯针

    def NeedleBarSolve(self):  # 针杆
        self.theta1_ab = self.theta1_0ab + self.n
        self.p1_b = p_rl(self.p1_a, self.l1_ab, self.theta1_ab, 0)
        self.p1_c = np.hstack([np.tile(self.p1_a[0:2], [self.n.size, 1]),
                               (self.p1_b[:, 2] - np.sqrt(self.l1_bc**2 - self.p1_b[:, 1]**2)).reshape(-1, 1)])
        self.p1_d = self.p1_c + np.array([0, 0, -self.l1_cd])
        self.p1_d1 = self.p1_d + np.array([self.l1_dd1, 0, 0])
        self.p1_d3 = self.p1_d + np.array([-self.l1_dd1, 0, 0])
        self.p1_e2 = self.p1_d + np.array([0, 0, -self.l1_de2])
        self.p1_e1 = self.p1_e2 + np.array([self.l1_dd1, 0, -self.l1_e1e2])
        self.p1_e3 = self.p1_e2 + np.array([-self.l1_dd1, 0, self.l1_e2e3])
        self.p1_f2 = self.p1_d + np.array([0, 0, -self.l1_df2])
        self.p1_f1 = self.p1_f2 + np.array([self.l1_dd1, 0, -self.l1_e1e2])
        self.p1_f3 = self.p1_f2 + np.array([-self.l1_dd1, 0, self.l1_e2e3])
        self.p1_g2 = self.p1_d + self.p1_g2_
        self.p1_g1 = self.p1_g2 + np.array([self.l1_dd1, 0, 0])
        self.p1_g3 = self.p1_g2 + np.array([-self.l1_dd1, 0, 0])
        self.p1_e1_top = self.p1_e1 + np.array([0, 0, 0.5])
        self.p1_e1_bottom = self.p1_e1 + np.array([0, 0, -0.5])
        self.p1_e2_top = self.p1_e2 + np.array([0, 0, 0.5])
        self.p1_e2_bottom = self.p1_e2 + np.array([0, 0, -0.5])
        self.p1_e3_top = self.p1_e3 + np.array([0, 0, 0.5])
        self.p1_e3_bottom = self.p1_e3 + np.array([0, 0, -0.5])

    def LooperSolve(self):  # 弯针
        self.p3_j_left_ = self.p3_k_ + np.array([-3.59986293, -0.03141553, 2.3])  # 弯针孔左边缘
        self.p3_j_bottom_ = self.p3_k_ + np.array([-3.99984769, -0.03490614, 1.9])  # 弯针孔下边缘
        self.p3_j_top_ = self.p3_k_ + np.array([-3.59986293, -0.03141553, 4])  # 弯针上边缘
        self.theta3_0ae = self.theta1_0ab + self.theta3_13
        self.theta3_0ab = self.theta3_0ae - self.theta3_bae
        self.theta3_0ab = self.theta3_0ab + self.n
        self.p3_b = p_rl(self.p3_a, self.l3_ab, self.theta3_0ab, 0)
        self.p3_c, self.theta3_0dc = rrr(self.p3_b[:, [1, 2]], self.p3_d[[1, 2]], self.l3_bc, self.l3_cd, -1)
        self.theta3_0ae = self.theta3_0ae + self.n
        self.p3_e = p_rl(self.p3_a, self.l3_ae, self.theta3_0ae, 0)
        l3_ef_xz = (np.sqrt(self.l3_ef**2 - (self.p3_e[:, 1] - self.p3_g[1])**2)).reshape(-1, 1)
        self.p3_f, self.theta3_0gf = rrr(self.p3_e[:, [0, 2]], self.p3_g[[0, 2]], l3_ef_xz, self.l3_fg, 1)
        self.theta3_0hi = self.theta3_0gf + self.theta3_fhi
        self.p3_i = p_rl(self.p3_h, self.l3_hi, self.theta3_0hi, 1)
        self.theta3_0dy = self.theta3_0dc + self.theta3_cdy
        self.p3_k = p_rp(self.p3_k_, self.theta3_0dy, 0) + np.hstack([0, self.p3_d[[1, 2]]])
        self.p3_j_left = p_rp(self.p3_j_left_, self.theta3_0dy, 0) + np.hstack([0, self.p3_d[[1, 2]]])
        self.p3_j_bottom = p_rp(self.p3_j_bottom_, self.theta3_0dy, 0) + np.hstack([0, self.p3_d[[1, 2]]])
        self.p3_j_top = p_rp(self.p3_j_top_, self.theta3_0dy, 0) + np.hstack([0, self.p3_d[[1, 2]]])
        self.p3_k = self.p3_k + np.hstack([self.p3_i[:, 0:1], np.tile([0, 0], [self.n.size, 1])])
        self.p3_j_left = self.p3_j_left + np.hstack([self.p3_i[:, 0:1], np.tile([0, 0], [self.n.size, 1])])
        self.p3_j_bottom = self.p3_j_bottom + np.hstack([self.p3_i[:, 0:1], np.tile([0, 0], [self.n.size, 1])])
        self.p3_j_top = self.p3_j_top + np.hstack([self.p3_i[:, 0:1], np.tile([0, 0], [self.n.size, 1])])

    def InvLooperSolve(self, p3_k_x, i):  # 弯针尖水平位置->上主轴角度
        p3_i_x = p3_k_x - self.p3_k_[0]
        theta3_0hi = np.arccos((p3_i_x - self.p3_h[0]) / self.l3_hi)
        theta3_0gf = theta3_0hi - self.theta3_fhi
        p3_f = p_rl(self.p3_g, self.l3_fg, theta3_0gf, 1)
        l3_ef_yz = (np.sqrt(self.l3_ef**2 - (self.p3_a[0] - p3_f[:, 0])**2))
        p3_e, theta3_0ae = rrr(p3_f[:, 1:3], self.p3_a[[1, 2]], l3_ef_yz, self.l3_ae, i)
        theta1_ab = theta3_0ae - self.theta3_13 - self.theta1_0ab
        return theta1_ab
