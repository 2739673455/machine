import numpy as np
import matplotlib.pyplot as plt
from function import *


def solve(p1, l1, theta1):
    du = 180 / np.pi
    hd = np.pi / 180
    n = np.linspace(0, 2 * np.pi, 361).reshape(-1, 1)

    theta1['0ab'] = n
    theta1['0ef'] = theta1['0ab'] - 95 * hd
    p1['b'] = p1['a'] + l1['ab'] * np.hstack((np.cos(theta1['0ab']) * np.sin(theta1['shangzhou']),
                                              np.cos(theta1['0ab']) * np.cos(theta1['shangzhou']),
                                              np.sin(theta1['0ab'])))
    p1['f'] = p1['e'] + l1['ef'] * np.hstack((np.cos(theta1['0ef']) * np.sin(theta1['shangzhou']),
                                              np.cos(theta1['0ef']) * np.cos(theta1['shangzhou']),
                                              np.sin(theta1['0ef'])))

    l1['fg1_xz'] = np.sqrt(l1['fg1']**2 - p1['f'][:, 1]**2)
    p1['g1'], theta1['0g2g1'] = rrr(p1['f'][:, [0, 2]], p1['g2'][[0, 2]], l1['fg1_xz'], l1['g1g2'], -1)
    p1['g1'] = np.insert(p1['g1'], 1, p1['g2'][1], axis=1)
    p1['g4'] = p_rl(np.array([48.11877609, 0, -0.35674442]), l1['g3g4'], np.pi + theta1['g2g3g4'], 1)
    p1['g4'] = p_rp(p1['g4'], theta1['0g2g1'], 1)

    p1['h1'], theta1['0h2h1'] = rrr(p1['g4'][:, [0, 2]], p1['h2'][[0, 2]], l1['g4h1'], l1['h1h2'], -1)
    p1['h1'] = np.insert(p1['h1'], 1, p1['g2'][1], axis=1)
    theta1['0h2h3'] = theta1['0h2h1'] + theta1['h1h2h3']
    theta1['0h3i1'] = theta1['0h2h3'] - np.pi + theta1['h2h3i1']
    p1['h3'] = p_rp(np.array([l1['h2h3'], 0, 0]), theta1['0h2h3'], 1) + p1['h2']
    p1['h5'] = p_rp(np.array([0, -4.99, 21.5]), theta1['0h2h3'], 1) + p1['h2']
    p1['o'] = p_rl(p1['n'], l1['no'], theta1['0no'], 1)

    l1['bc1_xz'] = np.sqrt(l1['bc1']**2 - (p1['b'][:, 1] - p1['h2_2'][1])**2).reshape(-1, 1)
    l1['c1c3'] = 113.50900801

    def func1(x):
        p1['c3'] = p_rl(p1['h5'][:, [0, 2]], l1['h5c3'], x)
        p1['c2'], theta1_0oc2 = rpr(p1['c3'], p1['o'][[0, 2]], l1['c2c3'], theta1['oc2c3'])
        p1['c1'], theta1_0bc1 = rrr(p1['c2'], p1['b'][:, [0, 2]], l1['c1c2'], l1['bc1_xz'], -1)
        _theta, l_target = theta_l(p1['c1'] - p1['c3'])
        return l_target
    x = np.tile(np.array([6 * hd]), (len(p1['h5']), 1))
    y = func1(x)
    d = np.hstack([np.tile(0, (len(y), 1)), np.sign(l1['c1c3'] - y) * hd, y, np.tile(l1['c1c3'], (len(y), 1))])
    theta1['0h5c3'], y2 = gradient(x, y, l1['c1c3'], d, func1)

    p1['c3'] = p_rl(p1['h5'][:, [0, 2]], l1['h5c3'], theta1['0h5c3'])
    p1['c2'], theta1['0oc2'] = rpr(p1['c3'], p1['o'][[0, 2]], l1['c2c3'], theta1['oc2c3'])
    p1['c1'], theta1['0bc1'] = rrr(p1['c2'], p1['b'][:, [0, 2]], l1['c1c2'], l1['bc1_xz'], -1)
    p1['c3'] = np.insert(p1['c3'], 1, p1['h2_2'][1], axis=1)
    p1['c2'] = np.insert(p1['c2'], 1, p1['h2_2'][1], axis=1)
    p1['c1'] = np.insert(p1['c1'], 1, p1['h2_2'][1], axis=1)
    p1['c4'] = p_rl(p1['c2'], l1['c2c4'], theta1['0oc2'] - np.pi, 1)
    p1['d'] = p_rp(p1['d'], theta1['0oc2'] - np.pi, 1) + p1['c4']
    p1['d2'] = p_rp(p1['d2'], theta1['0oc2'] - np.pi, 1) + p1['c4']
    p1['k1'] = p_rl(p1['j'], l1['jk1'], theta1['0jk1'], 1)
    theta1['0h2k1'], l1['k1h2'] = theta_l(p1['k1'][[0, 2]] - p1['h2'][[0, 2]])
    theta1['0h2k2'] = theta1['0h2k1'][0] + theta1['k1h2k2']
    p1['k2'] = p_rl(p1['h2_1'], l1['h2k2'], theta1['0h2k2'], 1)

    # 差动部分
    l1['i1i3'] = l1['i1i2'] + l1['i2i3']  # 93.3
    # 计算优化函数的左边界
    p1['i2_bound_left'], _p = rrp(p1['k2'][[0, 2]], p1['h3'][:, [0, 2]], l1['k2i2'], l1['i1i2'], theta1['0h3i1'], 90 * hd, 1)
    bound_left, _l = theta_l(p1['i2_bound_left'] - p1['k2'][[0, 2]])
    # 黄金分割法

    def func2(x):
        p1['i2'] = p_rl(p1['k2'][[0, 2]], l1['k2i2'], x)
        p1['i1_1'], _p = rrp(p1['i2'], p1['h3'][:, [0, 2]], l1['i1i2'], 0, theta1['0h3i1'], 0, 1)
        p1['i1_2'], _p = rrp(p1['i2'], p1['h3'][:, [0, 2]], l1['i1i2'], 0, theta1['0h3i1'], 0, -1)
        p1['i3'], _p = rrp(p1['i2'], p1['o'][[0, 2]], l1['i2i3'], l1['i3l1'], theta1['0oc2'], theta1['ol1i3'], -1)
        _theta, l_target1 = theta_l(p1['i1_1'] - p1['i3'])
        _theta, l_target2 = theta_l(p1['i1_2'] - p1['i3'])
        l_target = np.hstack([l_target1, l_target2])
        return np.min(abs(l_target - l1['i1i3']), axis=1).reshape(-1, 1)
    bound = np.hstack([bound_left + 1e-10, bound_left + 6 * hd])
    for i in range(30):
        x1 = 0.382 * np.diff(bound, axis=1) + bound[:, 0:1]
        x2 = 0.618 * np.diff(bound, axis=1) + bound[:, 0:1]
        i = (func2(x1) < func2(x2))
        bound = np.hstack([bound[:, 0:1] * i + x1 * ~i, x2 * i + bound[:, 1:2] * ~i])
    theta1['0k2i2'] = bound[:, 0:1]
    p1['i2'] = p_rl(p1['k2'][[0, 2]], l1['k2i2'], theta1['0k2i2'])
    p1['i1_1'], _p = rrp(p1['i2'], p1['h3'][:, [0, 2]], l1['i1i2'], 0, theta1['0h3i1'], 0, 1)
    p1['i1_2'], _p = rrp(p1['i2'], p1['h3'][:, [0, 2]], l1['i1i2'], 0, theta1['0h3i1'], 0, -1)
    p1['i3'], p1['l1'] = rrp(p1['i2'], p1['o'][[0, 2]], l1['i2i3'], l1['i3l1'], theta1['0oc2'], theta1['ol1i3'], -1)
    _theta, l_target1 = theta_l(p1['i1_1'] - p1['i3'])
    _theta, l_target2 = theta_l(p1['i1_2'] - p1['i3'])
    i = abs(l_target1 - l1['i1i3']) < abs(l_target2 - l1['i1i3'])
    p1['i1'] = p1['i1_1'] * i + p1['i1_2'] * ~i
    # _theta,l_target = theta_l(p1['i1']-p1['i3'])
    # print(abs(l_target-l1['i1i3']))
    p1['i2'] = np.insert(p1['i2'], 1, p1['h2_1'][1], axis=1)
    p1['i1'] = np.insert(p1['i1'], 1, p1['g2'][1], axis=1)
    p1['i3'] = np.insert(p1['i3'], 1, p1['g2'][1], axis=1)
    p1['l1'] = np.insert(p1['l1'], 1, p1['g2'][1], axis=1)
    p1['l2'] = p_rl(p1['l1'], l1['l1l2'], theta1['0oc2'] - np.pi, 1)
    p1['m'] = p_rp(p1['m'], theta1['0oc2'] - np.pi, 1) + p1['l2']
    p1['m2'] = p_rp(p1['m2'], theta1['0oc2'] - np.pi, 1) + p1['l2']
    p1['m3'] = p_rp(p1['m3'], theta1['0oc2'] - np.pi, 1) + p1['l2']

    print('针板前后:     ', 306.5, '-', 255.7, '  ', '针板高:', 10.8)
    print('差动牙水平位置:', np.max(p1['m'][:, 0]), '-', np.min(p1['m'][:, 0]), '=', np.ptp(p1['m'][:, 0]))
    print('送布牙水平位置:', np.max(p1['d'][:, 0]), '-', np.min(p1['d'][:, 0]), '=', np.ptp(p1['d'][:, 0]))
    print('差动牙竖直位置:', np.max(p1['m'][:, 2]), '-', np.min(p1['m'][:, 2]), '=', np.ptp(p1['m'][:, 2]))
    print('送布牙竖直位置:', np.max(p1['d'][:, 2]), '-', np.min(p1['d'][:, 2]), '=', np.ptp(p1['d'][:, 2]))

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.plot(p1['m'][:, 0], p1['m'][:, 2], 'g')
    ax.plot(p1['m2'][:, 0], p1['m2'][:, 2], 'g')
    ax.plot(p1['m3'][:, 0], p1['m3'][:, 2], 'g')
    ax.plot(p1['d'][:, 0], p1['d'][:, 2], 'b')
    ax.plot(p1['d2'][:, 0], p1['d2'][:, 2], 'b')
    ax.plot([255.7, 255.7, 306.5, 306.5], [8, 10.8, 10.8, 8])
    ax.plot([284.4, 284.4], [8, 10.8], '--')
    plt.show(block=False)


if __name__ == '__main__':
    from data import *
    solve(p1, l1, theta1)
