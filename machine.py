import numpy as np
from function import *


class Machine:
    def __init__(self):
        self.du = 180 / np.pi
        self.hd = np.pi / 180
        self.n = np.linspace(0, 359 * self.hd, 360).reshape(-1, 1)
        self.tph = -174.519  # Throat Plate height 针板高度
        # 机针
        self.p1_a = np.array([0, 0, 0])  # 上轴轴心
        self.p1_g2_ = np.array([0, 4, -17])  # 针夹头穿线孔位置
        self.l1_ab = 16.7  # 针杆曲柄
        self.l1_bc = 53  # 针杆连杆
        self.l1_cd = 77.919  # 针杆连杆到针夹头
        self.l1_dd1 = 2.8  # 相邻针间距
        self.l1_e1e2 = 2.4  # 左中针孔高低差
        self.l1_e2e3 = 2.7  # 中右针孔高低差
        self.l1_e2d = 43.7  # 中针孔到针夹头
        self.l1_f2d = 48.9  # 中针尖到针夹头
        self.theta1_0ab = 90 * self.hd  # 初始角度
        # 挑线
        self.p2_d = np.array([-116, 0, -31])  # 挑线轴轴心
        self.p2_f1_ = np.array([48.5, 41, 27.5])  # 挑线杆短孔左孔位置
        self.p2_g1_ = np.array([99, 41, 25.5])  # 挑线杆长孔左孔位置
        self.p2_h_ = np.array([50.91700975, 42.25, -13.22310427])  # 挑线杆下孔位置
        self.p2_i1 = np.array([5, 32.95758774, -50.52536094])  # 机壳上穿线孔左
        self.p2_i2 = np.array([0, 32.95758774, -50.52536094])  # 机壳上穿线孔中
        self.p2_i3 = np.array([-5, 32.95758774, -50.52536094])  # 机壳上穿线孔右
        self.p2_j1 = np.array([-135.53347593, 46.5148213, 42.5])  # 火柴杆左
        self.p2_j2 = np.array([-129.9043108, 49.7648213, 32.5])  # 火柴杆中
        self.p2_j3 = np.array([-124.27514568, 53.0148213, 22.5])  # 火柴杆右
        self.p2_k1 = np.array([-159, 58, 2])  # 硅油盒孔左
        self.p2_k2 = np.array([-159, 63, 2])  # 硅油盒孔中
        self.p2_k3 = np.array([-159, 68, 2])  # 硅油盒孔右
        self.l2_ab = 4.5  # 挑线偏心
        self.l2_bc = 30  # 挑线连杆
        self.l2_cd = 21  # 挑线摆杆
        self.l2_f1f2 = 5  # 挑线孔间距
        self.theta2_12 = 0 * self.hd  # 挑线偏心与主轴偏心角度
        self.theta2_cdx = 10 * self.hd  # 挑线杆与挑线摆杆角度
        # 弯针
        self.p3_a = np.array([-193.6, -28, -213.319])  # 主轴轴心
        self.p3_d = np.array([-193.6, 1.8, -198.319])  # 弯针滑杆轴心
        self.p3_g = np.array([-167, -31, -248.319])  # 弯针滑动曲柄轴心
        self.p3_h = np.array([-167, 1.8, -248.319])  # 弯针滑动摆杆轴心
        self.p3_j_ = np.array([160.33936845, -3, 17.60330317])  # 弯针孔
        self.p3_k_ = np.array([164.33936845, -2.09457126, 15.3])  # 弯针尖
        self.l3_ab = 1.3  # 弯针摆动偏心
        self.l3_bc = 25  # 弯针摆动连杆
        self.l3_cd = 18.48887640  # 弯针摆动摆杆
        self.l3_ae = 7.9  # 弯针滑动偏心
        self.l3_ef = 35  # 弯针滑动连杆
        self.l3_fg = 26.4  # 弯针滑动曲柄
        self.l3_hi = 53  # 弯针滑动摆杆
        self.theta3_13 = -3.05243756 * self.hd  # 机针曲柄与弯针滑动偏心角度
        self.theta3_bae = 130.5 * self.hd  # 弯针滑动偏心与摆动偏心角度
        self.theta3_fhi = -103.5 * self.hd  # 弯针滑动曲柄与滑动摆杆角度
        self.theta3_cdy = -161.5 * self.hd  # 弯针摆动摆杆与弯针架y方向角度
        # 送布
        self.p4_a = np.array([0, 0, 0])  # 主轴轴心
        self.p4_d2 = np.array([0, 13.50, -50.00]) + self.p4_a  # 针距调节轴
        self.p4_f = np.array([0, 34.00, -16.50]) + self.p4_a  # 送布牙摆动轴
        self.p4_t = np.array([0, 53.50, -12.50]) + self.p4_a  # 差动牙摆动轴
        self.p4_j = np.array([0, 53.50, 7.00]) + self.p4_a  # 右抬牙轴
        self.p4_x = np.array([0, 16, -47]) + self.p4_a  # 差动调节曲柄
        self.p4_l1_ = np.array([0, -12.8, 39.75])  # 送布牙齿坐标1
        self.p4_l2_ = np.array([0, 3.7, 39.75])  # 送布牙齿坐标2
        self.p4_r1_ = np.array([0, 5.25, 28.25])  # 差动牙齿坐标1
        self.p4_r2_ = np.array([0, 14.25, 28.25])  # 差动牙齿坐标2
        self.l4_ab3 = 3.8  # 送布偏心
        self.l4_b3c = 52
        self.l4_cd1 = 20
        self.l4_d1d2 = 20
        self.l4_ce = 36
        self.l4_ef = 14
        self.l4_fg = 15
        self.l4_gh = 13
        self.l4_fg2 = 18.5
        self.l4_g2v = 20
        self.l4_xw = 39.7
        self.l4_wv = 20
        self.l4_vu = 20
        self.l4_ut = 29.5
        self.l4_ts = 15  # 23,15
        self.l4_sh2 = 28
        self.l4_ab2 = 2.5  # 抬牙偏心
        self.l4_b2i = 53
        self.l4_ij = 15
        self.l4_jk = 7
        self.l4_kk2 = 7
        self.l4_ab1 = 1  # 主轴抬牙偏心
        self.l4_hb1_y = 1  # 送布牙架铰接点高度
        self.l4_h2b1_y = 10.5  # 差动牙架铰接点高度
        self.theta4_14 = -90 * self.hd  # 上轴与主轴抬牙偏心夹角
        self.theta4_0d2d3 = -13.03643004 * self.hd  # 针距
        self.theta4_d3d2d1 = 19.29045148 * self.hd
        self.theta4_efg = 143.7 * self.hd
        self.theta4_efg2 = -33.8 * self.hd
        self.theta4_0xw = -2 * self.hd  # 差动 小-25 ~ 大-2
        self.theta4_uts = -134.17943223 * self.hd  # -147.154682283,-134.17943223
        self.theta4_b1ab2 = -66.5 * self.hd  # 主轴抬牙偏心与抬牙偏心角度
        self.theta4_ijk = 72.3 * self.hd
        self.theta4_b1ab3 = -62.5 * self.hd  # 主轴抬牙偏心与送布偏心角度
        # 喂针
        self.p5_d = np.array([0, -38.6, -12])  # 喂针摆动连杆轴
        self.p5_f = np.array([-2, -38.6, -163.98174180])  # 喂针摆动架
        self.p5_h = np.array([-116, 56.80823691, -52.5])  # 喂针过线
        self.p5_i = np.array([-68.24929050, -41.49906339, -23])  # 喂针过线
        self.p5_j = np.array([-60.74929050, -41.49906339, -23])  # 喂针过线
        self.l5_ab = 2.5  # 喂针凸轮偏心
        self.l5_bc = 27  # 喂针连杆
        self.l5_cd = 29.5  # 喂针摆动曲柄
        self.l5_de = 43  # 喂针摆动连杆
        self.l5_ef = 16  # 喂针摆动架
        self.l5_fg = 47.91964548  # 喂针孔
        self.theta5_15 = -45 * self.hd
        self.theta5_cde = -63.27104462 * self.hd  # 喂针摆动曲柄和喂针摆动连杆夹角
        self.theta5_efg = -88.31011457 * self.hd  # 喂针摆动架和喂针夹角

    def Solve(self):
        self.NeedleBarSolve()  # 针杆
        self.ThreadPickSolve()  # 挑线
        self.LooperSolve()  # 弯针
        self.FabricFeedSolve()  # 送布
        self.FeedingNeedleSolve()  # 喂针

    def NeedleBarSolve(self):  # 针杆
        self.theta1_ab = self.theta1_0ab + self.n
        self.p1_b = p_rl(self.p1_a, self.l1_ab, self.theta1_ab, 0)
        self.p1_c = np.hstack([np.tile(self.p1_a[0:2], [self.n.size, 1]),
                               (self.p1_b[:, 2] - np.sqrt(self.l1_bc**2 - self.p1_b[:, 1]**2)).reshape(-1, 1)])
        self.p1_d = self.p1_c + np.array([0, 0, -self.l1_cd])
        self.p1_d1 = self.p1_d + np.array([self.l1_dd1, 0, 0])
        self.p1_d3 = self.p1_d + np.array([-self.l1_dd1, 0, 0])
        self.p1_e2 = self.p1_d + np.array([0, 0, -self.l1_e2d])
        self.p1_e1 = self.p1_e2 + np.array([self.l1_dd1, 0, -self.l1_e1e2])
        self.p1_e3 = self.p1_e2 + np.array([-self.l1_dd1, 0, self.l1_e2e3])
        self.p1_f2 = self.p1_d + np.array([0, 0, -self.l1_f2d])
        self.p1_f1 = self.p1_f2 + np.array([self.l1_dd1, 0, -self.l1_e1e2])
        self.p1_f3 = self.p1_f2 + np.array([-self.l1_dd1, 0, self.l1_e2e3])
        self.p1_g2 = self.p1_d + self.p1_g2_
        self.p1_g1 = self.p1_g2 + np.array([self.l1_dd1, 0, 0])
        self.p1_g3 = self.p1_g2 + np.array([-self.l1_dd1, 0, 0])

    def ThreadPickSolve(self):  # 挑线(左置)
        self.theta2_0ab = self.theta1_0ab + self.theta2_12 + self.n
        self.p2_b = p_rl(self.p1_a, self.l2_ab, self.theta2_0ab, 0)
        p2_c_z = (self.p2_b[:, 2] - np.sqrt(self.l2_bc**2 - (self.p2_b[:, 1] - self.p2_d[1])**2)).reshape(-1, 1)
        p2_c_x = self.p2_d[0] + np.sqrt(self.l2_cd**2 - (p2_c_z - self.p2_d[2])**2)
        self.p2_c = np.insert(np.hstack([p2_c_x, p2_c_z]), 1, self.p2_d[1], axis=1)
        self.p2_b[:, 0] = self.p2_c[:, 0]
        self.theta2_0dc = theta_l(self.p2_c[:, [0, 2]] - self.p2_d[[0, 2]])[0]
        self.theta2_0dx = self.theta2_0dc + self.theta2_cdx
        self.p2_f1 = p_rp(self.p2_f1_, self.theta2_0dx, 1) + np.array([self.p2_d[0], 0, self.p2_d[2]])
        self.p2_f2 = self.p2_f1 + np.array([0, self.l2_f1f2, 0])
        self.p2_f3 = self.p2_f1 + np.array([0, 2 * self.l2_f1f2, 0])
        self.p2_g1 = p_rp(self.p2_g1_, self.theta2_0dx, 1) + np.array([self.p2_d[0], 0, self.p2_d[2]])
        self.p2_g2 = self.p2_g1 + np.array([0, self.l2_f1f2, 0])
        self.p2_g3 = self.p2_g1 + np.array([0, 2 * self.l2_f1f2, 0])
        self.p2_h = p_rp(self.p2_h_, self.theta2_0dx, 1) + np.array([self.p2_d[0], 0, self.p2_d[2]])

    def LooperSolve(self):  # 弯针
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
        self.p3_j = p_rp(self.p3_j_, self.theta3_0dy, 0) + np.hstack([0, self.p3_d[[1, 2]]])
        self.p3_k = p_rp(self.p3_k_, self.theta3_0dy, 0) + np.hstack([0, self.p3_d[[1, 2]]])
        self.p3_j = self.p3_j + np.hstack([self.p3_i[:, 0:1], np.tile([0, 0], [self.n.size, 1])])
        self.p3_k = self.p3_k + np.hstack([self.p3_i[:, 0:1], np.tile([0, 0], [self.n.size, 1])])

    def FabricFeedSolve(self):  # 送布
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

        self.p4_v, self.theta4_0wv = rrr(self.p4_g2[:, 1:3], self.p4_w[[1, 2]], self.l4_g2v, self.l4_wv, -1)
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

        self.p4_h, _ = rrp(self.p4_g[:, 1:3], self.p4_b1[:, 1:3], self.l4_gh, self.l4_hb1_y, self.theta4_0b1k2, 90 * self.hd, -1)
        self.p4_h2, _ = rrp(self.p4_s[:, 1:3], self.p4_b1[:, 1:3], self.l4_sh2, self.l4_h2b1_y, self.theta4_0b1k2, -90 * self.hd, -1)
        self.p4_h = np.insert(self.p4_h, 0, 0, axis=1)
        self.p4_h2 = np.insert(self.p4_h2, 0, 0, axis=1)

        self.p4_l1 = p_rp(self.p4_l1_, self.theta4_0b1k2, 0) + self.p4_h
        self.p4_l2 = p_rp(self.p4_l2_, self.theta4_0b1k2, 0) + self.p4_h
        self.p4_r1 = p_rp(self.p4_r1_, self.theta4_0b1k2, 0) + self.p4_h2
        self.p4_r2 = p_rp(self.p4_r2_, self.theta4_0b1k2, 0) + self.p4_h2

    def FeedingNeedleSolve(self):  # 喂针
        self.theta5_0ab = self.theta1_0ab + self.theta5_15 + self.n
        self.p5_b = p_rl(self.p1_a, self.l5_ab, self.theta5_0ab, 0)
        self.p5_c, self.theta5_0dc = rrr(self.p5_b[:, 1:3], self.p5_d[[1, 2]], self.l5_bc, self.l5_cd, -1)
        self.theta5_0de = self.theta5_0dc + self.theta5_cde
        self.p5_e = p_rl(self.p5_d, self.l5_de, self.theta5_0de, 0)
        self.theta5_0fe = -np.arcsin((self.p5_e[:, 1] - self.p5_f[1]) / self.l5_ef) + np.pi
        self.theta5_0fg = self.theta5_0fe + self.theta5_efg
        self.p5_g = p_rl(self.p5_f, self.l5_fg, self.theta5_0fg.reshape(-1, 1), 2)

    def looperSynchronous(self):  # 计算弯针同步
        def getSynchronousAngle(self):  # 求对弯针同步时正反转两个角度
            machine_l1_ac_1 = self.l1_ab + self.l1_bc - 9  # 正转高9mm
            machine_l1_ac_2 = self.l1_ab + self.l1_bc - 9.8  # 反转高9.8mm
            synchronous_angle1 = np.arccos((self.l1_ab**2 + machine_l1_ac_1**2 - self.l1_bc**2) / (2 * self.l1_ab * machine_l1_ac_1))  # 正转角度
            synchronous_angle2 = np.arccos((self.l1_ab**2 + machine_l1_ac_2**2 - self.l1_bc**2) / (2 * self.l1_ab * machine_l1_ac_2))  # 反转角度
            return np.array([[synchronous_angle1 + np.pi, -synchronous_angle2 + np.pi]]).T

        def getLooperDistance(x, self):  # 两角度下弯针尖距离
            self.theta3_13 = x[0] * self.hd
            self.Solve()
            return (self.p3_k[0][0] - self.p3_k[1][0])**2

        self.n = getSynchronousAngle(self)
        x0 = [-3]
        result = minimize(getLooperDistance, x0, args=self, tol=1e-15)
        print(result)
        print(result.x)

    def invLooperSolve(self, p3_k_x: np.ndarray, i):  # 弯针尖水平位置->上主轴角度
        p3_i_x = p3_k_x - self.p3_k_[0]
        theta3_0hi = np.arccos((p3_i_x - self.p3_h[0]) / self.l3_hi)
        theta3_0gf = theta3_0hi - self.theta3_fhi
        p3_f = p_rl(self.p3_g, self.l3_fg, theta3_0gf, 1)
        l3_ef_yz = (np.sqrt(self.l3_ef**2 - (self.p3_a[0] - p3_f[:, 0])**2))
        p3_e, theta3_0ae = rrr(p3_f[:, 1:3], self.p3_a[[1, 2]], l3_ef_yz, self.l3_ae, i)
        theta1_ab = theta3_0ae - self.theta3_13 - self.theta1_0ab
        return theta1_ab


class Cam:  # 凸轮
    def __init__(self):
        self.du = 180 / np.pi
        self.hd = np.pi / 180
        self.cam_info = np.array([
            [0.37469953, -22.52725562, 5.35666683, 1],
            [-3.592602, -17.51691194, 11.74752165, 1],
            [27.34200794, -22.2547628, 19.54780265, -1],
            [-18.35758299, 5.90520844, 34.1312118, 1],
            [-1.81936149, 5.95113971, 17.59292652, 1],
            [-2.87455758, 4.95936565, 19.04104756, 1],
            [-3.29784457, 3.56316712, 20.5, 1],
            [-1.98960972, 2.82932684, 22.00, 1],
            [3.51379771, 2.54177587, 27.51091453, 1],
            [105.03265005, 37.47574683, 134.87226561, 1],
            [-0.85372899, -7.74952254, 19.73213248, 1],
            [2.02724392, 1.03948428, 28.98127546, 1]])  # 圆弧圆心，圆弧半径，法向,最后一位是0为直线

    def SolveTangencyPoint(self):  # 求圆弧切点坐标
        cam_len = len(self.cam_info)
        index1 = np.array([cam_len - 1])
        index1 = np.append(index1, np.arange(0, cam_len - 1))
        index2 = np.arange(0, cam_len)
        rate = (self.cam_info[index1, 2] * self.cam_info[index1, 3] /
                (self.cam_info[index1, 2] * self.cam_info[index1, 3] - self.cam_info[index2, 2] * self.cam_info[index2, 3])).reshape(-1, 1)
        tangency_points = (self.cam_info[index2, 0:2] - self.cam_info[index1, 0:2]) * rate + self.cam_info[index1, 0:2]

        if 0 in self.cam_info[:, 3]:  # cam_info最后一位是0为直线
            straight_index = np.arange(cam_len)[self.cam_info[:, 3] == 0]
            straight_index_previous = straight_index - 1
            straight_index_next = straight_index + 1
            tangency_points[straight_index] = self.cam_info[straight_index][:, 0:2] * self.cam_info[straight_index_previous][:, 2:3] + self.cam_info[straight_index_previous][:, 0:2]
            tangency_points[straight_index_next] = self.cam_info[straight_index][:, 0:2] * self.cam_info[straight_index_next][:, 2:3] + self.cam_info[straight_index_next][:, 0:2]

        self.tangency_points = tangency_points

    def SolveAngleRange(self):  # 求圆弧角度范围
        self.SolveTangencyPoint()
        angles = theta_l(self.tangency_points)[0] * self.du
        angles[angles < 0] += 360
        self.angle_range = np.hstack([angles, np.vstack([angles[1:, 0:1], angles[0, 0]])])
        if self.angle_range[0, 0] == 0:
            self.angle_range[-1, -1] = 360

    def SetCamOutline(self):  # 凸轮轮廓
        self.SolveAngleRange()
        self.arc_c = self.cam_info[:, 0:2]  # 凸轮圆弧圆心
        self.arc_r = self.cam_info[:, 2:3]  # 凸轮圆弧半径
        self.arc_t = self.cam_info[:, 3:4]  # 凸轮圆弧法向
        c_theta, l0 = theta_l(self.arc_c)  # 凸轮圆弧圆心位置对应角度
        angle_range_round = np.ceil(self.angle_range)  # 起始角度取整
        cam_p = np.array([[0, 0]])
        for i, val in enumerate(angle_range_round):
            if self.cam_info[i, 3] == 0:  # 直线部分直接添加交点
                cam_p = np.vstack([cam_p, self.tangency_points[i]])
            else:
                if val[0] > val[1]:
                    cam_theta = np.hstack([np.arange(val[0], 360), np.arange(0, val[1])]) * self.hd
                else:
                    cam_theta = np.arange(val[0], val[1]) * self.hd
                alpha = cam_theta - c_theta[i]
                beta = np.zeros(alpha.shape)
                beta[alpha != 0] = np.arcsin(l0[i] / self.arc_r[i] * np.sin(alpha[alpha != 0]))
                gamma = cam_theta + beta if self.arc_t[i] == 1 else cam_theta + np.pi - beta
                gamma = gamma.reshape(-1, 1)
                cam_p_current = self.arc_c[i] + self.arc_r[i] * np.hstack([np.cos(gamma), np.sin(gamma)])
                cam_p = np.vstack([cam_p, cam_p_current])
        self.p = cam_p[1:]

    def Rotate(self, theta, *args):
        if type(theta) is np.ndarray:
            if len(args):
                self.p = p_rp(self.p, theta, args[0])
                self.tangency_points = p_rp(self.tangency_points, theta, args[0])
            else:
                self.p = p_rp(self.p, theta)
                self.tangency_points = p_rp(self.tangency_points, theta)
            self.arc_c = p_rp(self.arc_c, theta)
        else:
            if len(args):
                self.p = p_rp(self.p, theta, args[0])[0]
                self.tangency_points = p_rp(self.tangency_points, theta, args[0])[0]
            else:
                self.p = p_rp(self.p, theta)[0]
                self.tangency_points = p_rp(self.tangency_points, theta)[0]
            self.arc_c = p_rp(self.arc_c, theta)[0]


class Trough:  # 过线槽
    def __init__(self):
        self.du = 180 / np.pi
        self.hd = np.pi / 180
        self.arc_r = 1.5
        self.arc_c = np.array([[17.5, 0], [-17.5, 0]])

    def SetTroughOutline(self):
        particle_theta = np.linspace(-90, 90, 10).reshape(-1, 1) * self.hd
        self.p = np.hstack([np.cos(particle_theta), np.sin(particle_theta)]) * self.arc_r + self.arc_c[0]
        particle_theta = particle_theta + np.pi
        self.p = np.vstack([self.p, np.hstack([np.cos(particle_theta), np.sin(particle_theta)]) * self.arc_r + self.arc_c[1]])

    def Rotate(self, theta, *args):
        if len(args):
            self.p = p_rp(self.p, theta, args[0])[0]
        else:
            self.p = p_rp(self.p, theta)[0]
        self.arc_c = p_rp(self.arc_c, theta)[0]


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    k6 = Machine()
    k6.Solve()
    cam1 = Cam()
    trough1 = Trough()
