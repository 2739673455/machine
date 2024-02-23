import sys
sys.path.append('./..')
import numpy as np
from function import *
from data import dahe, k6
from machine import Machine
import matplotlib.pyplot as plt

k6.p2_g1_ = np.array([103, 0, 25.5])  # 挑线杆长孔位置
k6.Solve()


def setInterval(speed):  # 转过1°时间间隔
    return 1 / (speed / 60 * 360)


def needleBarVelocity(machine):
    velocity = np.diff(machine.p1_d, axis=0)[:, 2]
    return velocity / interval


def threadPickVelocity(machine):
    diff_val = np.diff(machine.p2_g1, axis=0)
    velocity_sign = (diff_val[:, 2] > 0) * 2 - 1
    velocity = np.sqrt(np.sum(diff_val**2, axis=1)) * velocity_sign
    return velocity / interval


interval = setInterval(4000)
k6_needlebar_velocity = needleBarVelocity(k6)
dahe_needlebar_velocity = needleBarVelocity(dahe)
k6_threadpick_velocity = threadPickVelocity(k6)
dahe_threadpick_velocity = threadPickVelocity(dahe)


plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(k6_needlebar_velocity, ':r', linewidth=0.7, label='杰克-机针速度')
ax1.plot(dahe_needlebar_velocity, ':g', linewidth=0.7, label='大和-机针速度')
ax1.plot(k6_threadpick_velocity, 'r', linewidth=0.7, label='杰克-挑线速度')
ax1.plot(dahe_threadpick_velocity, 'g', linewidth=0.7, label='大和-挑线速度')
ax1.set_xlabel('°', fontsize=12)
ax1.set_ylabel('mm/s', fontsize=12)
ax1.legend()
ax1.grid()
plt.show()
