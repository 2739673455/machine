import sys
sys.path.append('./..')
import numpy as np
from function import *
from machine import Machine


class Machine1(Machine):
    def Solve(self):
        self.NeedleBarSolve()  # 针杆
        self.ThreadPickSolve()  # 挑线

    def NeedleBarSolve(self):  # 针杆
        super(Machine1, self).NeedleBarSolve()
        self.p1_h1 = np.array([-self.l1_dd1, 0, self.tph])
        self.p1_h2 = np.array([0, 0, self.tph])
        self.p1_h3 = np.array([self.l1_dd1, 0, self.tph])


class Machine2(Machine):
    def Solve(self):
        self.NeedleBarSolve()  # 针杆
        self.ThreadPickSolve()  # 挑线

    def NeedleBarSolve(self):  # 针杆
        super(Machine2, self).NeedleBarSolve()
        self.p1_h1 = np.array([-self.l1_dd1, 0, self.tph])
        self.p1_h2 = np.array([0, 0, self.tph])
        self.p1_h3 = np.array([self.l1_dd1, 0, self.tph])

    def ThreadPickSolve(self):  # 挑线(右置)
        self.theta2_0ab = self.theta1_0ab + self.theta2_12 + self.n
        self.p2_b = p_rl(self.p1_a, self.l2_ab, self.theta2_0ab, 0)
        p2_c_z = (self.p2_b[:, 2] - np.sqrt(self.l2_bc**2 - (self.p2_b[:, 1] - self.p2_d[1])**2)).reshape(-1, 1)
        p2_c_x = self.p2_d[0] - np.sqrt(self.l2_cd**2 - (p2_c_z - self.p2_d[2])**2)
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


dahe = Machine1()
dahe.tph = -171.7  # Throat Plate height 针板高度
# 针杆
dahe.p1_a = np.array([0, 0, 0])  # 上轴轴心
dahe.p1_g2_ = np.array([0, 3.35702099, -16.65120672])  # 针杆座穿线孔位置
dahe.l1_ab = 16.5  # 针杆曲柄
dahe.l1_bc = 53  # 针杆连杆
dahe.l1_cd = 74.7  # 针杆连杆到针夹头
dahe.l1_dd1 = 2.8  # 相邻两针间距
dahe.l1_e1e2 = 2.6  # 左中针孔高低差
dahe.l1_e2e3 = 2.5  # 中右针孔高低差
dahe.l1_e2d = 43.15  # 中针孔到针夹头
dahe.l1_f2d = 48.5  # 中针尖到针夹头
dahe.theta1_0ab = 90 * dahe.hd  # 初始角度
# 挑线
dahe.p2_d = np.array([-104, -1.32, -30])  # 挑线轴轴心
dahe.p2_f1_ = np.array([48, 38.60036513, 28])  # 挑线杆短孔位置
dahe.p2_g1_ = np.array([99, 42.39109265, 23.5])  # 挑线杆长孔位置
dahe.p2_i1 = np.array([6, 35.37118183, -38.60759007])  # 机壳上穿线孔左
dahe.p2_i2 = np.array([2, 35.37118183, -38.60759007])  # 机壳上穿线孔中
dahe.p2_i3 = np.array([-2, 35.37118183, -38.60759007])  # 机壳上穿线孔右
dahe.p2_j1 = np.array([-127.40510436, 42.79573269, 42.25])  # 火柴杆左
dahe.p2_j2 = np.array([-122.43087893, 46.15089011, 34.25])  # 火柴杆中
dahe.p2_j3 = np.array([-117.45665349, 49.50604753, 25])  # 火柴杆右
dahe.l2_ab = 4.7  # 挑线偏心
dahe.l2_bc = 33  # 挑线连杆
dahe.l2_cd = 21.5  # 挑线摆杆
dahe.l2_f1f2 = 3.5  # 挑线孔间距
dahe.theta2_12 = 0 * dahe.hd  # 针杆曲柄与挑线偏心角度
dahe.theta2_cdx = 21.05273501 * dahe.hd  # 挑线杆与挑线摆杆角度
# 喂针
dahe.p5_d = np.array([0, -38.5, -12])  # 喂针摆动连杆轴
dahe.p5_f = np.array([-2, -38.6, -163.98174180])  # 喂针摆动架
dahe.p5_h = np.array([-116, 56.80823691, -52.5])  # 喂针过线
dahe.p5_i = np.array([-68.24929050, -41.49906339, -23])  # 喂针过线
dahe.p5_j = np.array([-60.74929050, -41.49906339, -23])  # 喂针过线
dahe.l5_ab = 2.2  # 喂针凸轮偏心
dahe.l5_bc = 29  # 喂针连杆
dahe.l5_cd = 30.01594583  # 喂针摆动曲柄
dahe.l5_de = 44.5  # 喂针摆动连杆
dahe.l5_ef = 16  # 喂针摆动连杆
dahe.l5_fg = 47.91964548
dahe.theta5_15 = -45 * dahe.hd
dahe.theta5_cde = -63.27104462 * dahe.hd  # 喂针摆动曲柄和喂针摆动连杆夹角
dahe.theta5_efg = -88.31011457 * dahe.hd  # 喂针摆动架和喂针夹角
dahe.Solve()


k6 = Machine1()
k6.tph = -174.369  # Throat Plate height 针板高度
# 机针
k6.p1_a = np.array([0, 0, 0])  # 上轴轴心
k6.p1_g2_ = np.array([0, 3.35702099, -16.65120672])  # 针夹头穿线孔位置
k6.l1_ab = 16.7  # 针杆曲柄
k6.l1_bc = 53  # 针杆连杆
k6.l1_cd = 77.769  # 针杆连杆到针夹头
k6.l1_dd1 = 2.8  # 相邻针间距
k6.l1_e1e2 = 2.4  # 左中针孔高低差
k6.l1_e2e3 = 2.7  # 中右针孔高低差
k6.l1_e2d = 43.55  # 中针孔到针夹头
k6.l1_f2d = 48.9  # 中针尖到针夹头
k6.theta1_0ab = 90 * k6.hd  # 初始角度
# 挑线
k6.p2_d = np.array([-104, -0.7, -31])  # 挑线轴轴心
k6.p2_f1_ = np.array([48.5, 40.29809725, 27.5])  # 挑线杆短孔位置
k6.p2_g1_ = np.array([99, 40.29809725, 25.5])  # 挑线杆长孔位置
k6.p2_i1 = np.array([5, 31.98601031, -51.88482262])  # 机壳上穿线孔左
k6.p2_i2 = np.array([0, 31.98601031, -51.88482262])  # 机壳上穿线孔中
k6.p2_i3 = np.array([-5, 31.98601031, -51.88482262])  # 机壳上穿线孔右
k6.p2_j1 = np.array([-124.53347593, 46.5148213, 43])  # 火柴杆左
k6.p2_j2 = np.array([-118.9043108, 49.7648213, 33])  # 火柴杆中
k6.p2_j3 = np.array([-113.27514568, 53.0148213, 23])  # 火柴杆右
k6.l2_ab = 4.5  # 挑线偏心
k6.l2_bc = 30  # 挑线连杆
k6.l2_cd = 21  # 挑线摆杆
k6.l2_f1f2 = 5  # 挑线孔间距
k6.theta2_12 = 0 * k6.hd  # 挑线偏心与主轴偏心角度
k6.theta2_cdx = 9.57146814 * k6.hd  # 挑线杆与挑线摆杆角度

# k6.theta2_12 = 180 * k6.hd  # 挑线偏心与主轴偏心角度
# k6.theta2_cdx = -164.790017634486 * k6.hd  # 挑线杆与挑线摆杆角度

# k6.Solve()


k6_2 = Machine2()  # 挑线轴位置116，右置
k6_2.tph = -174.369  # Throat Plate height 针板高度
# 机针
k6_2.p1_a = np.array([0, 0, 0])  # 上轴轴心
k6_2.p1_g2_ = np.array([0, 3.35702099, -16.65120672])  # 针夹头穿线孔位置
k6_2.l1_ab = 16.7  # 针杆曲柄
k6_2.l1_bc = 53  # 针杆连杆
k6_2.l1_cd = 77.769  # 针杆连杆到针夹头
k6_2.l1_dd1 = 2.8  # 相邻针间距
k6_2.l1_e1e2 = 2.4  # 左中针孔高低差
k6_2.l1_e2e3 = 2.7  # 中右针孔高低差
k6_2.l1_e2d = 43.55  # 中针孔到针夹头
k6_2.l1_f2d = 48.9  # 中针尖到针夹头
k6_2.theta1_0ab = 90 * k6_2.hd  # 初始角度
# 挑线
k6_2.p2_d = np.array([-116, -0.7, -31])  # 挑线轴轴心
k6_2.p2_f1_ = np.array([48.5, 40.29809725, 27.5])  # 挑线杆短孔位置
k6_2.p2_g1_ = np.array([99, 40.29809725, 25.5])  # 挑线杆长孔位置
k6_2.p2_i1 = np.array([5, 31.98601031, -51.88482262])  # 机壳上穿线孔左
k6_2.p2_i2 = np.array([0, 31.98601031, -51.88482262])  # 机壳上穿线孔中
k6_2.p2_i3 = np.array([-5, 31.98601031, -51.88482262])  # 机壳上穿线孔右
k6_2.p2_j1 = np.array([-135.53347593, 46.5148213, 43])  # 火柴杆左
k6_2.p2_j2 = np.array([-129.9043108, 49.7648213, 33])  # 火柴杆中
k6_2.p2_j3 = np.array([-124.27514568, 53.0148213, 23])  # 火柴杆右
k6_2.l2_ab = 4.5  # 挑线偏心
k6_2.l2_bc = 30  # 挑线连杆
k6_2.l2_cd = 21  # 挑线摆杆
k6_2.l2_f1f2 = 5  # 挑线孔间距
k6_2.theta2_12 = 180 * k6_2.hd  # 挑线偏心与主轴偏心角度
k6_2.theta2_cdx = -164.790017634486 * k6_2.hd  # 挑线杆与挑线摆杆角度
k6_2.Solve()
