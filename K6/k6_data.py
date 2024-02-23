import numpy as np
import pandas as pd

p = dict()
l = dict()
theta = dict()
hd = np.pi / 180
du = 180 / np.pi


def k6_data(p, l, theta):
    ThroatPlate_height = -174.519
    l['tph'] = ThroatPlate_height

    # 针杆
    p['1_a'] = np.array([0, 0, 0])
    p['1_g2'] = np.array([0, 4, -17])  # 针杆座孔位置
    l['1_ab'] = 16.7  # 针杆曲柄
    l['1_bc'] = 53  # 针杆连杆
    l['1_cd'] = 78.619  # 针杆连杆到针杆座
    l['1_dd1'] = 2.8  # 针间距
    l['1_e1e2'] = 2.4  # 针孔高低差
    l['1_e2d'] = 43.7  # 中针孔到针杆座
    l['1_f2d'] = 48.9  # 中针尖到针杆座
    theta['1_0ab'] = 90 * hd  # 初始角度

    # 挑线
    p['2_a'] = np.array([0, 0, 0])  # 轴心
    p['2_d'] = np.array([-116, 0, -31])  # 挑线轴轴心
    p['2_e'] = np.array([-116, 41, -31])  # 挑线杆旋转中心
    p['2_f1'] = np.array([49.5, 0, 27.5])  # 短杆孔位置
    p['2_g1'] = np.array([100, 0, 25.5])  # 长杆孔位置
    p['2_h'] = np.array([51.91320445, 1.25, -13.31026001])  # 下杆孔位置
    p['2_i1'] = np.array([5, 32.95758774, -50.52536094])  # 机壳上穿线孔左
    p['2_i2'] = np.array([0, 32.95758774, -50.52536094])  # 机壳上穿线孔中
    p['2_i3'] = np.array([-5, 32.95758774, -50.52536094])  # 机壳上穿线孔右
    p['2_j1'] = np.array([-135.53347593, 46.5148213, 42.5])  # 火柴杆左
    p['2_j2'] = np.array([-129.9043108, 49.7648213, 32.5])  # 火柴杆中
    p['2_j3'] = np.array([-124.27514568, 53.0148213, 22.5])  # 火柴杆右
    p['2_k1'] = np.array([-159, 58, 2])  # 硅油盒孔左
    p['2_k2'] = np.array([-159, 63, 2])  # 硅油盒孔中
    p['2_k3'] = np.array([-159, 68, 2])  # 硅油盒孔右
    l['2_ab'] = 4.5  # 挑线偏心
    l['2_bc'] = 30  # 挑线连杆
    l['2_cd'] = 21  # 挑线摆杆
    l['2_f1f2'] = 5  # 挑线孔间距
    l['2_f1f3'] = 10  # 挑线孔间距
    theta['2_cex'] = 10 * hd  # 挑线杆与挑线摆杆角度

    # 弯针
    p['3_a'] = np.array([-193, -28, -213.319])  # 弯针滑动偏心中心
    p['3_d'] = np.array([-193, 1.8, -198.319])  # 弯针滑杆轴心
    p['3_g'] = np.array([-167, -31, -248.319])  # 弯针滑动曲柄轴心
    p['3_h'] = np.array([-167, 1.8, -248.319])  # 弯针滑动摆杆轴心
    p['3_j'] = np.array([159.31137048, -3, 17.60330317])  # 弯针孔
    p['3_k'] = np.array([163.31137048, -2.09457126, 15.3])  # 弯针尖
    l['3_ab'] = 1.3  # 弯针摆动偏心
    l['3_bc'] = 25  # 弯针摆动连杆
    l['3_cd'] = 16.96732153  # 弯针摆动摆杆
    l['3_ae'] = 7.9  # 弯针滑动偏心
    l['3_ef'] = 35  # 弯针滑动连杆
    l['3_fg'] = 26.4  # 弯针滑动曲柄
    l['3_hi'] = 53  # 弯针滑动摆杆
    theta['3_13'] = -2.9 * hd  # 机针曲柄与弯针滑动偏心角度
    theta['3_bae'] = 130.5 * hd  # 弯针滑动偏心与摆动偏心角度
    theta['3_fhi'] = -104.65497429 * hd  # 弯针滑动曲柄与滑动摆杆角度
    theta['3_cdx'] = -162.6840332 * hd  # 弯针摆动摆杆与弯针角度
    p['3_wanzhen_1'] = np.array([[136, -3.4, 19.3],
                                 [136, -4.2, 17.6],
                                 [136, -4.2, 16.8],
                                 [136, -2.4, 16.8],
                                 [136, -2.4, 19.3]])
    p['3_wanzhen_2'] = np.array([[156, -3.4, 19.3],
                                 [156, -4.2, 17.6],
                                 [156, -4.2, 16.8],
                                 [156, -2.4, 16.8],
                                 [156, -2.4, 19.3]])

    # 送布
    p['4_a'] = np.array([0, -28, -213.319])
    p['4_f'] = np.array([0, 6, -229.819])
    p['4_v'] = np.array([0, 25.5, -225.819])
    p['4_m'] = np.array([0, 25.5, -206.319])
    l['4_ab'] = 4
    l['4_bc'] = 52
    l['4_cd'] = 20
    l['4_ce'] = 36
    l['4_ef'] = 14
    l['4_fg'] = 15
    l['4_fg2'] = 18.5
    l['4_g2t'] = 20
    l['4_tz'] = 20
    l['4_ut'] = 20
    l['4_vw'] = 23
    l['4_ak'] = 1.75
    l['4_kl'] = 53
    l['4_lm'] = 15
    l['4_mn'] = 10
    l['4_ai'] = 1.2
    l['4_gh'] = 13
    l['4_hj'] = 1
    l['4_wh2'] = 28
    l['4_h2j2'] = 10.5
    l['4_no'] = 7
    theta['4_14'] = -46.19469349 * hd
    theta['4_kab'] = 14.5 * hd
    theta['4_kai'] = 54 * hd
    theta['4_efg'] = 145 * hd
    theta['4_efg2'] = -28.6182638 * hd
    theta['4_uvw'] = -142 * hd
    theta['4_lmn'] = 71 * hd

    # 打线凸轮
    theta_cam = 0  # 凸轮索引0对应角度为:第0个点角度+theta_cam
    cam = pd.DataFrame(data=None, columns=['theta0', 'c', 'r', 't'])  # 起始角度，圆心，半径，法向
    cam.loc[0] = [np.array([270, 277.88180317]), np.array([0.37469953, -22.52725562]), 5.35666683, 1]
    cam.loc[1] = [np.array([277.88180317, 292.56866309]), np.array([-3.592602, -17.51691194]), 11.74752165, 1]
    cam.loc[2] = [np.array([292.56866309, 311.72232405]), np.array([27.34200794, -22.2547628]), 19.54780265, -1]
    cam.loc[3] = [np.array([311.72232405, 20.82605858]), np.array([-18.35758299, 5.90520844]), 34.1312118, 1]
    cam.loc[4] = [np.array([20.82605858, 58.57043439]), np.array([-1.81936149, 5.95113971]), 17.59292652, 1]
    cam.loc[5] = [np.array([58.57043439, 97.12501635]), np.array([-2.87455758, 4.95936565]), 19.04104756, 1]
    cam.loc[6] = [np.array([97.12501635, 132.73578327]), np.array([-2.5279201, 1.00484528]), 23.00, 1]
    cam.loc[7] = [np.array([132.73578327, 170.57495762]), np.array([-1.98960972, 2.82932684]), 22.00, 1]
    cam.loc[8] = [np.array([170.57495762, 195.90146059]), np.array([3.51379771, 2.54177587]), 27.51091453, 1]
    cam.loc[9] = [np.array([195.90146059, 219.20720351]), np.array([105.03265005, 37.47574683]), 134.87226561, 1]
    cam.loc[10] = [np.array([219.20720351, 255.20323776]), np.array([-0.85372899, -7.74952254]), 19.73213248, 1]
    cam.loc[11] = [np.array([255.20323776, 270]), np.array([2.02724392, 1.03948428]), 28.98127546, 1]
    cam_c = np.vstack(cam['c'])
    i = ((cam_c[:, 1] < 0) * -2 + 1).reshape(-1, 1)
    l0 = np.sqrt(np.sum(cam_c**2, axis=1)).reshape(-1, 1)  # 凸轮中心到圆心距离
    c_theta = np.zeros(l0.shape)
    j = (l0 != 0).reshape(-1)  # 凸轮中心不与圆心重合的
    c_theta[j] = np.arccos(cam_c[j][:, 0:1] / l0[j]) * i[j]  # 凸轮圆心位置对应角度
    cam['c_theta'] = c_theta
    cam['l0'] = l0
    cam['theta'] = cam['theta0'].apply(lambda theta: np.ceil(theta))  # 起始角度取整
    cam_p = np.array([[0, 0]])
    for i, val in enumerate(cam['theta']):
        if val[0] > val[1]:
            cam.at[i, 'theta'] = np.hstack([np.arange(val[0], 360), np.arange(0, val[1])]) * hd
        else:
            cam.at[i, 'theta'] = np.arange(val[0], val[1]) * hd
        alpha = cam.at[i, 'theta'] - cam.loc[i, 'c_theta']
        beta = np.zeros(alpha.shape)
        beta[alpha != 0] = np.arcsin(cam.loc[i, 'l0'] / cam.loc[i, 'r'] * np.sin(alpha[alpha != 0]))
        theta1 = cam.at[i, 'theta'] + beta if cam.loc[i, 't'] == 1 else cam.loc[i, 'theta'] + np.pi - beta
        theta1 = theta1.reshape(-1, 1)
        r = np.hstack([np.cos(theta1), np.sin(theta1)])
        p_cam = cam.loc[i, 'c'] + cam.loc[i, 'r'] * r
        cam_p = np.vstack([cam_p, p_cam])
    cam_p = cam_p[1:]
    cam_p = np.vstack([cam_p[theta_cam:], cam_p[:theta_cam]])
    p['cam_info'] = cam
    p['cam'] = cam_p
    p['5_a'] = np.array([-214.441, 66.367, -209.069])  # 凸轮中心
    p['5_b1'] = np.array([-203.740, 81.532, -200.073])  # 左穿线孔，直径2
    p['5_b2'] = np.array([-224.940, 81.532, -200.073])  # 右穿线孔，直径2
    # 中间过线槽
    p5_c_theta = np.linspace(-90, 90, 10).reshape(-1, 1) * hd
    p['5_trough'] = np.hstack([1.5 * np.cos(p5_c_theta), 1.5 * np.sin(p5_c_theta)]) + np.array([17.5, 0])
    p5_c_theta = p5_c_theta + np.pi
    p['5_trough'] = np.vstack([p['5_trough'], np.hstack([1.5 * np.cos(p5_c_theta), 1.5 * np.sin(p5_c_theta)]) + np.array([-17.5, 0])])
    p['5_trough_center'] = np.array([[17.5, 0], [-17.5, 0]])


if __name__ == '__main__':
    k6_data(p, l, theta)
