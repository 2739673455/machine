import numpy as np

p = dict()
l = dict()
theta = dict()
hd = np.pi / 180
du = 180 / np.pi


def k6_data2(p, l, theta):
    ThroatPlate_height = 38.8
    l['tph'] = ThroatPlate_height

    # 送布
    p['4_a'] = np.array([0, 0, 0])
    p['4_d2'] = np.array([0, 13.50, -50.00])
    p['4_f'] = np.array([0, 34.00, -16.50])
    p['4_t'] = np.array([0, 53.50, -12.50])
    p['4_j'] = np.array([0, 53.50, 7.00])
    p['4_y'] = np.array([0, 57.00, -53.00])
    p['4_x3'] = np.array([0, 16, -47])

    l['4_ab3'] = 3.8
    l['4_b3c'] = 52
    l['4_cd1'] = 20
    l['4_d1d2'] = 20

    l['4_ce'] = 36
    l['4_ef'] = 14
    l['4_fg'] = 15
    l['4_gh'] = 13

    l['4_fg2'] = 26
    l['4_g2v'] = 14

    l['4_yx1'] = 27.7
    l['4_x1x2'] = 35
    l['4_x2x3'] = 25
    l['4_x3w'] = 34
    l['4_wv'] = 19
    l['4_vu'] = 19
    l['4_ut'] = 25.5
    l['4_ts'] = 23  # 23,15
    l['4_sh2'] = 28

    l['4_ab2'] = 2.5
    l['4_b2i'] = 53
    l['4_ij'] = 15
    l['4_jk'] = 7
    l['4_kk2'] = 7

    l['4_ab1'] = 1
    l['4_h2'] = 10.5

    theta['4_0d2d3'] = -13.03643004 * hd  # 针距 小12.22303614 ~ 大-13.03643004
    theta['4_d3d2d1'] = 19.29045148 * hd
    theta['4_efg'] = 143.7 * hd
    theta['4_efg2'] = -53.8 * hd  # -53.8*hd

    theta['4_0yx1'] = -126.59457802 * hd  # 差动 小-126.59457802 ~ 大-103.0815767
    theta['4_x2x3w'] = 87.3 * hd
    theta['4_uts'] = 181.60492451 * hd  # 168.22228562*hd,181.60492451*hd

    theta['4_ijk'] = 72.3 * hd
    theta['4_b1ab2'] = -66.5 * hd
    theta['4_b1ab3'] = -62.5 * hd


if __name__ == '__main__':
    k6_data2(p, l, theta)
