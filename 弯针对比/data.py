import sys
sys.path.append(sys.path[0] + '\..')
import numpy as np
from machine import Machine

dahe = Machine()
dahe.p3_a = np.array([-198.5, -27.5, -210.7])  # 主轴轴心
dahe.p3_d = np.array([-198.5, 1.5, -195.5])  # 弯针滑杆轴心
dahe.p3_g = np.array([-172.5, -27.5, -246.2])  # 弯针滑动曲柄轴心
dahe.p3_h = np.array([-172.5, 1.5, -246.2])  # 弯针滑动摆杆轴心
dahe.p3_k_ = np.array([171.46837602309992, -1.78510440, 16.3])  # 弯针尖
dahe.l3_ab = 1.45  # 弯针摆动偏心
dahe.l3_bc = 25  # 弯针摆动连杆
dahe.l3_cd = 19.58671281  # 弯针摆动摆杆
dahe.l3_ae = 7.9  # 弯针滑动偏心
dahe.l3_ef = 35.5  # 弯针滑动连杆
dahe.l3_fg = 26.5  # 弯针滑动曲柄
dahe.l3_hi = 53.5  # 弯针滑动摆杆
# dahe.theta3_13 = 0  # 机针曲柄与弯针滑动偏心角度
dahe.theta3_bae = 120 * dahe.hd  # 弯针滑动偏心与摆动偏心角度
dahe.theta3_fhi = -102 * dahe.hd  # 弯针滑动曲柄与滑动摆杆角度
dahe.theta3_cdy = -161.24743731 * dahe.hd  # 弯针摆动摆杆与弯针架y方向角度
dahe.Solve()

k6_3mm = Machine()
k6_3mm.l3_cd = 18  # 弯针摆动摆杆
k6_3mm.theta3_cdy = -162.00156651 * k6_3mm.hd  # 弯针摆动摆杆与弯针架y方向角度
# k6_3mm.theta3_0ae = 265.98752101 * k6_3mm.hd  # 弯针在最右端时，弯针滑动曲柄角度
k6_3mm.Solve()

k6_0mm = Machine()
k6_0mm.p3_g = np.array([-167, -28, -248.319])  # 弯针滑动曲柄轴心
k6_0mm.p3_k_ = np.array([164.55711446290402, -2.09457126, 15.3])  # 弯针尖
k6_0mm.theta3_fhi = -103.5 * k6_0mm.hd  # 弯针滑动曲柄与滑动摆杆角度
k6_0mm.l3_cd = 18.5  # 弯针摆动摆杆
k6_0mm.theta3_cdy = -161.83597746851012 * k6_0mm.hd  # 弯针摆动摆杆与弯针架y方向角度
k6_0mm.Solve()
