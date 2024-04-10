import sys
sys.path.append(sys.path[0] + '\..')
import numpy as np
import pandas as pd
from function import *
from pandas import Series, DataFrame

du = 180 / np.pi
hd = np.pi / 180
n = np.linspace(0, 359 * hd, 360).reshape(-1, 1)
ONE = np.ones([len(n), 1])


def solve(p, l, theta, n):

    # 针杆
    theta['1_ab'] = theta['1_0ab'] + n
    p['1_b'] = p_rl(p['1_a'], l['1_ab'], theta['1_ab'], 0)
    p['1_c'] = np.hstack([ONE * p['1_a'][0], ONE * p['1_a'][1], (p['1_b'][:, 2] - np.sqrt(l['1_bc']**2 - p['1_b'][:, 1]**2)).reshape(-1, 1)])
    p['1_d'] = p['1_c'] - np.array([0, 0, l['1_cd']])
    p['1_d1'] = p['1_d'] + np.array([l['1_dd1'], 0, 0])
    p['1_d3'] = p['1_d'] - np.array([l['1_dd1'], 0, 0])
    p['1_e2'] = p['1_d'] + np.array([0, 0, -l['1_e2d']])
    p['1_e1'] = p['1_e2'] + np.array([l['1_dd1'], 0, -l['1_e1e2']])
    p['1_e3'] = p['1_e2'] - np.array([l['1_dd1'], 0, -l['1_e1e2']])
    p['1_f2'] = p['1_d'] + np.array([0, 0, -l['1_f2d']])
    p['1_f1'] = p['1_f2'] + np.array([l['1_dd1'], 0, -l['1_e1e2']])
    p['1_f3'] = p['1_f2'] - np.array([l['1_dd1'], 0, -l['1_e1e2']])
    p['1_g2'] = p['1_d'] + p['1_g2']
    p['1_g1'] = p['1_g2'] + np.array([l['1_dd1'], 0, 0])
    p['1_g3'] = p['1_g2'] - np.array([l['1_dd1'], 0, 0])

    # 挑线
    theta['2_0ab'] = theta['1_0ab'] + 0 * hd + n
    p['2_b'] = p_rl(p['2_a'], l['2_ab'], theta['2_0ab'], 0)
    p['2_c'] = np.hstack([ONE * 0, ONE * p['1_a'][0], (p['2_b'][:, 2] - np.sqrt(l['2_bc']**2 - p['2_b'][:, 1]**2)).reshape(-1, 1)])
    p['2_c'][:, 0] = p['2_d'][0] + np.sqrt(l['2_cd']**2 - (p['2_c'][:, 2] - p['2_d'][2])**2)
    p['2_b'][:, 0] = p['2_c'][:, 0]
    theta['2_0dc'] = theta_l(p['2_c'][:, [0, 2]] - p['2_d'][[0, 2]])[0]
    theta['2_0ex'] = theta['2_0dc'] + theta['2_cex']
    p['2_f1'] = p_rp(p['2_f1'], theta['2_0ex'], 1) + p['2_e']
    p['2_f2'] = p['2_f1'] + np.array([0, l['2_f1f2'], 0])
    p['2_f3'] = p['2_f1'] + np.array([0, l['2_f1f3'], 0])
    p['2_g1'] = p_rp(p['2_g1'], theta['2_0ex'], 1) + p['2_e']
    p['2_g2'] = p['2_g1'] + np.array([0, l['2_f1f2'], 0])
    p['2_g3'] = p['2_g1'] + np.array([0, l['2_f1f3'], 0])
    p['2_h'] = p_rp(p['2_h'], theta['2_0ex'], 1) + p['2_e']

    # 弯针
    theta['3_0ae'] = theta['1_0ab'] + theta['3_13']
    theta['3_0ab'] = theta['3_0ae'] - theta['3_bae']
    theta['3_ab'] = theta['3_0ab'] + n
    p['3_b'] = p_rl(p['3_a'], l['3_ab'], theta['3_ab'], 0)
    p['3_c'], theta['3_0dc'] = rrr(p['3_b'][:, [1, 2]], p['3_d'][[1, 2]], l['3_bc'], l['3_cd'], -1)
    p['3_c'] = np.hstack([p['3_b'][:, 0:1], p['3_c']])
    theta['3_ae'] = theta['3_0ae'] + n
    p['3_e'] = p_rl(p['3_a'], l['3_ae'], theta['3_ae'], 0)
    l3_ef_xz = (np.sqrt(l['3_ef']**2 - (p['3_e'][:, 1] - p['3_g'][1])**2)).reshape(-1, 1)
    p['3_f'], theta['3_0gf'] = rrr(p['3_e'][:, [0, 2]], p['3_g'][[0, 2]], l3_ef_xz, l['3_fg'], 1)
    p['3_f'] = np.insert(p['3_f'], 1, p['3_g'][1], axis=1)
    theta['3_0hi'] = theta['3_0gf'] + theta['3_fhi']
    p['3_i'] = p_rl(p['3_h'], l['3_hi'], theta['3_0hi'], 1)
    theta['3_0dx'] = theta['3_0dc'] + theta['3_cdx']
    p['3_j'] = p_rp(p['3_j'], theta['3_0dx'], 0) + np.hstack([0, p['3_d'][[1, 2]]])
    p['3_k'] = p_rp(p['3_k'], theta['3_0dx'], 0) + np.hstack([0, p['3_d'][[1, 2]]])
    p['3_j'][:, 0] = p['3_j'][:, 0] + p['3_i'][:, 0]
    p['3_k'][:, 0] = p['3_k'][:, 0] + p['3_i'][:, 0]
    p['3_wanzhen_1'] = p_rp(p['3_wanzhen_1'], theta['3_0dx'], 0) + \
        np.array([np.hstack([p['3_i'][:, 0:1], ONE * p['3_d'][1], ONE * p['3_d'][2]])]).transpose(1, 0, 2)
    p['3_wanzhen_2'] = p_rp(p['3_wanzhen_2'], theta['3_0dx'], 0) + \
        np.array([np.hstack([p['3_i'][:, 0:1], ONE * p['3_d'][1], ONE * p['3_d'][2]])]).transpose(1, 0, 2)

    # 送布
    theta['4_0ak'] = theta['1_0ab'] + theta['4_14']
    theta['4_zhenju'] = -17.58907668 * hd
    p['4_d'] = p_rl(np.array([0, -14.5, -263.319]), 20, theta['4_zhenju'] + 26.5 * hd, 0)
    theta['4_chadong'] = 22.66820466 * hd
    p['4_z'] = p_rl(np.array([0, 29, -266.319]), 27.7, theta['4_chadong'] - 150 * hd, 0)
    theta['4_0z'] = rrr(p['4_z'][[1, 2]], np.array([-12, -260.319]), 24, 25, -1)[1].reshape(-1)
    p['4_z'] = p_rl(np.array([0, -12, -260.319]), 39.7, theta['4_0z'] + 85.66621286 * hd, 0)
    theta['4_ak'] = theta['4_0ak'] + n
    theta['4_ab'] = theta['4_ak'] + theta['4_kab']
    theta['4_ai'] = theta['4_ak'] + theta['4_kai']
    p['4_b'] = p_rl(p['4_a'], l['4_ab'], theta['4_ab'], 0)
    p['4_c'], theta['4_0dc'] = rrr(p['4_b'][:, [1, 2]], p['4_d'][[1, 2]], l['4_bc'], l['4_cd'], 1)
    p['4_c'] = np.hstack([p['4_b'][:, 0:1], p['4_c']])
    p['4_e'], theta['4_0fe'] = rrr(p['4_c'][:, [1, 2]], p['4_f'][[1, 2]], l['4_ce'], l['4_ef'], 1)
    p['4_e'] = np.hstack([p['4_c'][:, 0:1], p['4_e']])
    theta['4_0fg'] = theta['4_0fe'] + theta['4_efg']
    p['4_g'] = p_rl(p['4_f'], l['4_fg'], theta['4_0fg'], 0)
    theta['4_0fg2'] = theta['4_0fe'] + theta['4_efg2']
    p['4_g2'] = p_rl(p['4_f'], l['4_fg2'], theta['4_0fg2'], 0)
    p['4_t'] = rrr(p['4_g2'][:, [1, 2]], p['4_z'][[1, 2]], l['4_g2t'], l['4_tz'], -1)[0]
    p['4_t'] = np.hstack([p['4_g2'][:, 0:1], p['4_t']])
    theta['4_0vt'], l['4_tv'] = theta_l(p['4_t'][:, [1, 2]] - p['4_v'][[1, 2]])
    theta['4_tvu'] = -np.arccos((27.5**2 + l['4_tv']**2 - l['4_ut']**2) / (2 * 27.5 * l['4_tv']))
    theta['4_0vw'] = theta['4_0vt'] + theta['4_tvu'] + theta['4_uvw']
    p['4_w'] = p_rl(p['4_v'], l['4_vw'], theta['4_0vw'], 0)
    p['4_k'] = p_rl(p['4_a'], l['4_ak'], theta['4_ak'], 0)
    p['4_l'], theta['4_0ml'] = rrr(p['4_k'][:, [1, 2]], p['4_m'][[1, 2]], l['4_kl'], l['4_lm'], -1)
    theta['4_0mn'] = theta['4_0ml'] + theta['4_lmn']
    p['4_n'] = p_rl(p['4_m'], l['4_mn'], theta['4_0mn'], 0)
    p['4_i'] = p_rl(p['4_a'], l['4_ai'], theta['4_ai'], 0)
    theta['4_0in'], l['4_in'] = theta_l(p['4_n'][:, [1, 2]] - p['4_i'][:, [1, 2]])
    theta['4_nin2'] = -np.arcsin(l['4_no'] / l['4_in'])
    theta['4_0in2'] = theta['4_0in'] + theta['4_nin2']
    p['4_h'], p['4_j'] = rrp(p['4_g'][:, [1, 2]], p['4_i'][:, [1, 2]], l['4_gh'], l['4_hj'], theta['4_0in2'], 90 * hd, -1)
    p['4_h2'], p['4_j2'] = rrp(p['4_w'][:, [1, 2]], p['4_i'][:, [1, 2]], l['4_wh2'], l['4_h2j2'], theta['4_0in2'], -90 * hd, -1)
    p['4_h'] = np.hstack([p['4_g'][:, 0:1], p['4_h']])
    p['4_h2'] = np.hstack([p['4_w'][:, 0:1], p['4_h2']])
    p['4_j'] = np.hstack([p['4_g'][:, 0:1], p['4_j']])
    p['4_j2'] = np.hstack([p['4_w'][:, 0:1], p['4_j2']])
    p['4_q'] = p_rp(np.array([0, -13.2, 38.6]), theta['4_0in2'], 0) + p['4_j']
    p['4_q_2'] = p_rp(np.array([0, 4.5, 38.6]), theta['4_0in2'], 0) + p['4_j']
    p['4_q2'] = p_rp(np.array([0, 12.35, 38.6]), theta['4_0in2'], 0) + p['4_j2']
    p['4_q2_2'] = p_rp(np.array([0, 3.25, 38.6]), theta['4_0in2'], 0) + p['4_j2']

    # 线环
    p['arg_1_f1_directionchange'] = np.argwhere(np.diff(p['1_f1'][:, 2]) > 0)[0, 0]  # 机针变向时刻
    p['arg_1_e1_tph_intersect'] = np.argwhere(abs(p['1_e1'][:, 2] - l['tph']) < 0.1).reshape(-1)  # 机针孔过针板时刻
    p['arg_1_f1_3_k_intersect'] = np.argwhere(abs(p['1_e1'][:, 0] - p['3_k'][:, 0]) < 0.2).reshape(-1)  # 弯针过机针孔时刻

    p1_e1_1 = p['1_e1'][p['arg_1_f1_directionchange']:p['arg_1_e1_tph_intersect'][1]]
    num_curve1 = p['arg_1_f1_3_k_intersect'][1] - p['arg_1_f1_directionchange']
    num_curve2 = p['arg_1_e1_tph_intersect'][1] - p['arg_1_f1_3_k_intersect'][1]
    num_curve3 = num_curve1 + num_curve2

    p['thread_point1'] = p1_e1_1

    point2_y = np.hstack([np.linspace(0, -3, num_curve1), np.linspace(-3, -3, num_curve2)]).reshape(-1, 1)
    point2_z = np.hstack([np.linspace(0, -1, num_curve1), np.linspace(-1, -1, num_curve2)]).reshape(-1, 1)
    p['thread_point2'] = np.hstack([np.zeros(point2_y.shape), point2_y, point2_z]) + p1_e1_1

    point3_y = np.hstack([np.linspace(0, -6, num_curve1), np.linspace(-6, -6, num_curve2)]).reshape(-1, 1)
    point3_z = np.hstack([np.linspace(0, 2, num_curve1), np.linspace(2, 2, num_curve2)]).reshape(-1, 1)
    p['thread_point3'] = np.hstack([np.zeros(point3_y.shape), point3_y, point3_z]) + p1_e1_1

    p['thread_point4'] = np.hstack([p1_e1_1[:, 0:1], np.zeros(point3_y.shape) - 0.4, np.ones(point3_z.shape) * l['tph']])

    p['ring'] = Series(dtype='float64')

    for i in range(p['arg_1_e1_tph_intersect'][0]):  # 机针下降
        p['ring'][str(i)] = np.vstack([p['2_i1'],
                                       p['1_g1'][i],
                                       p['1_e1'][i]])

    for i in range(p['arg_1_e1_tph_intersect'][0], p['arg_1_f1_directionchange']):  # 机针孔过针板
        p['ring'][str(i)] = np.vstack([p['2_i1'],
                                       p['1_g1'][i],
                                       np.array([p['1_e1'][i, 0], p['1_e1'][i, 1], l['tph']]),
                                       p['1_e1'][i],
                                       np.array([p['1_e1'][i, 0], p['1_e1'][i, 1] - 0.4, l['tph']])])

    for i in range(p['arg_1_f1_directionchange'], p['arg_1_f1_3_k_intersect'][1]):  # 线环形成
        j = i - p['arg_1_f1_directionchange']
        p['ring'][str(i)] = np.insert(
            np.vstack([p['2_i1'][1:],
                       np.array([p['1_g1'][i, 1], p['1_g1'][i, 2]]),
                       np.array([p['1_e1'][0, 1], l['tph']]),
                       bezierCurve(np.vstack([p['thread_point1'][j, [1, 2]],
                                              p['thread_point2'][j, [1, 2]],
                                              p['thread_point3'][j, [1, 2]],
                                              p['thread_point4'][j, [1, 2]]]))]),
            0, p['1_e1'][0, 0], axis=1)

    for i in range(p['arg_1_f1_3_k_intersect'][1], p['arg_1_e1_tph_intersect'][1]):  # 弯针勾线
        j = i - p['arg_1_f1_directionchange']
        p['ring'][str(i)] = np.insert(
            np.vstack([p['2_i1'][1:],
                       np.array([p['1_g1'][i, 1], p['1_g1'][i, 2]]),
                       np.array([p['1_e1'][0, 1], l['tph']]),
                       interpolateTransform(
                p['ring'][str(p['arg_1_f1_3_k_intersect'][1] - 1)][2:, [1, 2]],
                boundaryLookup(np.vstack([p['1_e1'][i, [1, 2]], np.array([-0.4, l['tph']])]), p['3_wanzhen_1'][i][:, [1, 2]])[1:],
                num_curve3 - num_curve1, j - num_curve1)
            ]),
            0, p['1_e1'][0, 0], axis=1)

    for i in range(p['arg_1_e1_tph_intersect'][1], len(n)):  # 机针孔出针板
        j = i - p['arg_1_f1_directionchange']
        p['ring'][str(i)] = np.insert(
            np.vstack([p['2_i1'][1:],
                       np.array([p['1_g1'][i, 1], p['1_g1'][i, 2]]),
                       boundaryLookup(np.vstack([np.array([p['1_e1'][0, 1], l['tph']]), np.array([-0.4, l['tph']])]),
                                      p['3_wanzhen_1'][i][:, [1, 2]])]),
            0, p['1_e1'][0, 0], axis=1)

    l['length1'] = np.zeros(n.shape)
    for i in range(len(n)):
        l['length1'][i] = stringLength(p['ring'][i])


if __name__ == '__main__':
    from k6_data import *
    from k6_plot import *
    k6_data(p, l, theta)
    solve(p, l, theta, n)
    k6_plot(p, l, theta, n)
    # print(p['arg_1_f1_directionchange'])
    # print(p['arg_1_e1_tph_intersect'])
    # print(p['arg_1_f1_3_k_intersect'])
