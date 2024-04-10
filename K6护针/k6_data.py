import sys
sys.path.append(sys.path[0] + '\..')
from function import *
from machine import Machine


class K6(Machine):
    def Solve(self):
        self.NeedleBarSolve()  # 针杆
        # self.ThreadPickSolve()  # 挑线
        self.LooperSolve()  # 弯针
        # self.FabricFeedSolve()  # 送布
        self.HuZhenSolve()  # 护针

    def HuZhenSolve(self):  # 护针
        l5_e1e2_xz = 2.8 - self.p5_e1[0]
        self.p5_e2 = np.array([2.8,
                               self.p5_e1[1] + l5_e1e2_xz * np.tan(self.theta5_xy),
                               self.p5_e1[2] + l5_e1e2_xz * np.tan(self.theta5_xz1)])
        self.p5_f2 = np.array([2.8,
                               self.p5_f1[1] + l5_e1e2_xz * np.tan(self.theta5_xy),
                               self.p5_f1[2] + l5_e1e2_xz * np.tan(self.theta5_xz2)])

        self.theta5_0ab = self.theta1_0ab + self.theta5_15 + self.n
        self.p5_b = p_rl(self.p5_a, self.l5_ab, self.theta5_0ab, 0)
        self.theta5_0cb, _ = theta_l(self.p5_b[:, 1:3] - self.p5_c[1:3])
        self.theta5_0cd = self.theta5_0cb + self.theta5_bcd
        self.p5_e2 = p_rp(self.p5_e2, self.theta5_0cd, 0) + self.p5_c
        self.p5_f2 = p_rp(self.p5_f2, self.theta5_0cd, 0) + self.p5_c


k6 = K6()
# 机针
k6.p1_a = np.array([0, 0, 0])  # 上轴轴心
k6.p1_g2_ = np.array([0, 4, -17])  # 针杆座穿线孔位置
k6.l1_ab = 16.7  # 针杆曲柄
k6.l1_bc = 53  # 针杆连杆
k6.l1_cd = 79.269  # 针杆连杆到针杆座
k6.l1_dd1 = 2.8  # 相邻针间距
k6.l1_e1e2 = 2.4  # 左中针孔高低差
k6.l1_e2e3 = 2.7  # 中右针孔高低差
k6.l1_e2d = 43.7  # 中针孔到针杆座
k6.l1_f2d = 48.9  # 中针尖到针杆座
k6.theta1_0ab = 90 * k6.hd  # 初始角度
# 弯针
k6.p3_a = np.array([-193, -28, -215.169])  # 主轴轴心
k6.p3_d = np.array([-193, 1.8, -200.169])  # 弯针滑杆轴心
k6.p3_g = np.array([-167, -31, -250.169])  # 弯针滑动曲柄轴心
k6.p3_h = np.array([-167, 1.8, -250.169])  # 弯针滑动摆杆轴心
k6.p3_j_ = np.array([160.30152435, -3, 17.60330317])  # 弯针孔
k6.p3_k_ = np.array([163.99866731, -2.09457126, 15.3])  # 弯针尖
k6.l3_ab = 1.1  # 弯针摆动偏心
k6.l3_bc = 25  # 弯针摆动连杆
k6.l3_cd = 18.32957856  # 弯针摆动摆杆
k6.l3_ae = 7.9  # 弯针滑动偏心
k6.l3_ef = 35  # 弯针滑动连杆
k6.l3_fg = 26.4  # 弯针滑动曲柄
k6.l3_hi = 53  # 弯针滑动摆杆
k6.theta3_13 = 2.94 * k6.hd  # 机针曲柄与弯针滑动偏心角度
k6.theta3_bae = 130.5 * k6.hd  # 弯针滑动偏心与摆动偏心角度
k6.theta3_fhi = -104.02 * k6.hd  # 弯针滑动曲柄与滑动摆杆角度
k6.theta3_cdy = -160.61997401 * k6.hd  # 弯针摆动摆杆与弯针架y方向角度
# 护针
k6.p5_a = np.array([0, -28, -215.169])  # 主轴
k6.p5_c = np.array([0, -13, -223.169])  # 护针曲柄轴心
k6.p5_e1 = np.array([-5.42558263, 13.2, 33.07094805])
k6.p5_f1 = np.array([-5.42558263, 13.2, 32.37094805])
k6.l5_ab = 0.58
k6.l5_e_xy = 9.8
k6.theta5_15 = 74.97233645 * k6.hd
k6.theta5_bcd = -150.43342514 * k6.hd
k6.theta5_xz1 = -12.53367695 * k6.hd
k6.theta5_xz2 = -22 * k6.hd
k6.theta5_xy = -6 * k6.hd
