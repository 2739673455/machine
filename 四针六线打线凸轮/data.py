import sys
sys.path.append('./..')
import numpy as np
from machine import Cam
from function import *


class ThreadPickRod():
    def __init__(self):
        self.du = 180 / np.pi
        self.hd = np.pi / 180
        self.n = np.linspace(0, 359 * self.hd, 360).reshape(-1, 1)
        self.p1_a = np.array([0, 0, 0])
        self.p1_d = np.array([0, 51.4, 11])
        self.l1_ab = 9.8
        self.l1_bc = 40
        self.l1_cd = 33.67625276
        self.theta1_cdf = 129.02158159 * self.hd

        self.below_side1 = np.array([[0, 0, -4], [0, 23.83254199, -4], [3.5, 31.33831622, -4]])
        self.below_side2 = np.array([[1.5, 0, -4], [1.5, 23.5, -4], [5, 31.00577422, -4]])

        self.n = self.n - 26.21477644 * self.hd

    def Solve(self):
        self.SolveTheta()
        self.SolveBelowSide()

    def SolveTheta(self):  # 计算挑线杆摆动角度
        self.p1_b = p_rl(self.p1_a, self.l1_ab, self.n, 0)
        self.p1_c, self.theta1_0dc = rrr(self.p1_b[:, 1:3], self.p1_d[1:3], self.l1_bc, self.l1_cd, 1)
        self.theta1_0df = self.theta1_0dc + self.theta1_cdf

    def SolveBelowSide(self):  # 计算下边线摆动后坐标
        self.below_side1 = p_rp(self.below_side1, self.theta1_0df, 0)
        self.below_side2 = p_rp(self.below_side2, self.theta1_0df, 0)
        self.vector_below = np.diff(self.below_side1[:, 0:2], axis=-2)
        self.vector_below = np.vstack(self.vector_below)
        self.normal = self.vector_below / np.sqrt(np.sum(self.vector_below**2, axis=1)).reshape(-1, 1)
        self.normal_t = p_rp(self.normal, -np.pi / 2, 0)[0]


class Cam_a(Cam):
    def __init__(self):
        self.du = 180 / np.pi
        self.hd = np.pi / 180
        self.cam_info = np.array([
            [0, -1, 21.8, 0],
            [11.87749965, -18.80, 3, 1],
            [3, -10, 15.5, 1],
            [0.94551858, 0.32556815, 15.08, 0],
            [8.99527306, 4.83708929, 5, 1],
            [0.76604444, 0.64278761, 15, 0],
            [2.17128167, 12.96960553, 5, 1],
            [-0.5817214, -3.29910597, 21.5, 1],
            [-9.06589635, -13.8, 8, 1]
        ])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rod1 = ThreadPickRod()
    rod1.Solve()
    cam1 = Cam_a()
    cam1.SetCamOutline()
