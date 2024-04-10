import numpy as np
from scipy.special import comb
from scipy.optimize import root, minimize
# import shapely
# from shapely.geometry import Point, LineString, Polygon


def modulus(p):  # 求向量的模
    return np.sqrt(np.sum(p**2, axis=-1))


def stringLength(string):  # 串长度
    string = np.diff(string, axis=0)
    length = np.sum(modulus(string))
    return length


def theta_l(p):  # 求向量与横轴正方向夹角(-180°~180°)与向量的模 二维
    p = p.reshape(1, -1) if p.ndim == 1 else p
    sign = ((p[:, 1] < 0) * -2 + 1).reshape(-1, 1)
    l = modulus(p).reshape(-1, 1)
    theta = np.zeros(l.shape)
    nonzero = (l != 0).reshape(-1)
    theta[nonzero] = np.arccos(p[nonzero][:, 0:1] / l[nonzero]) * sign[nonzero]
    return theta, l


def includedAngle(p1, p2):  # 两向量夹角，p1 逆时针转向 p2(0°~360°)  ,三维张量
    p1 = p1.reshape(1, -1) if p1.ndim == 1 else p1
    p2 = p2.reshape(1, -1) if p2.ndim == 1 else p2
    l1 = modulus(p1)
    l2 = modulus(p2)
    p1_dot_p2 = np.sum(p1 * p2, axis=-1)
    cos_val = p1_dot_p2 / l1 / l2
    angle_sign = (np.cross(p1, p2) < 0)
    index_above_1 = abs(cos_val) > 1  # 浮点数误差导致cosθ值大于1
    cos_val[index_above_1] = np.sign(cos_val[index_above_1]) * np.floor(abs(cos_val[index_above_1]))
    theta = np.arccos(cos_val)
    theta[angle_sign] = 2 * np.pi - theta[angle_sign]
    return theta


def p_rl(p0, l, theta, *args):  # 起点为p0,长度为l的杆，以args[0]轴为旋转中心,旋转θrad,args:x=0,y=1,z=2
    r = np.hstack([np.cos(theta), np.sin(theta)])
    if len(args):  # 不带args为二维坐标，带args为三维坐标
        r = np.insert(r, args[0], 0, axis=1) if r.ndim >= 2 else np.insert(r, args[0], 0)
    return p0 + l * r


def p_rp(p1, theta, *args):  # 坐标为p1的点，以args[0]轴为旋转中心,旋转θrad,args:x=0,y=1,z=2
    theta = theta.reshape(-1) if type(theta) is np.ndarray else theta
    r = np.array([np.vstack([np.cos(theta), -np.sin(theta)]), np.vstack([np.sin(theta), np.cos(theta)])])
    if len(args):  # 不带args为二维坐标，带args为三维坐标
        r = np.insert(
            np.array([np.insert(r[0], args, 0, axis=0), np.insert(r[1], args, 0, axis=0)]),
            args,
            np.insert(np.zeros(r[0].shape),
                      args,
                      1,
                      axis=0),
            axis=0)
    p2 = p1 @ r
    p2 = p2.T
    return p2


def rrr(p2, p4, l23, l34, i=1):  # 转动副-转动副-转动副,二维，i=1/-1(theta243正,负)
    try:
        l23 = l23.reshape(-1, 1)
    except:
        pass
    theta042, l24 = theta_l(p2 - p4)
    theta243 = np.arccos((l34**2 + l24**2 - l23**2) / 2 / l34 / l24) * i
    theta043 = (theta042 + theta243).reshape(-1, 1)
    p3 = p_rl(p4, l34, theta043)
    return p3, theta043


def rrp(p2, p5, l23, l34, theta054, theta543, i=1):  # 转动副-转动副-移动副,二维,i=1/-1(33_方向)
    p6 = p_rl(p5, l34, theta054 + theta543 + np.pi)
    theta062, l26 = theta_l(p2 - p6)
    a1 = 1
    a2 = -2 * l26 * np.cos(theta062 - theta054)
    a3 = l26**2 - l23**2
    a = np.concatenate((np.tile(np.array([a1]), a3.shape), a2, a3), axis=1)
    l36 = np.array(list(map(lambda x: np.roots(x), a)))
    if i == 1:
        l36 = np.max(l36, axis=1).reshape(-1, 1)
    else:
        l36 = np.min(l36, axis=1).reshape(-1, 1)
    l36[~np.isreal(l36)] = np.nan
    p3 = p_rl(p6, l36, theta054)
    p4 = p_rl(p5, l36, theta054)
    return p3, p4


def rpr(p2, p4, l23, theta432, i=1):  # 转动副-移动副-转动副,二维,i=1/-1(方向),l12<l14+l23有多解
    theta024, l24 = theta_l(p4 - p2)
    a1 = 1
    a2 = -2 * l23 * np.cos(theta432)
    a3 = l23**2 - l24**2
    a = np.concatenate((np.tile(np.array([a1]), a3.shape),
                        np.tile(np.array([a2]), a3.shape),
                        a3), axis=1)
    l34 = np.array(list(map(lambda x: np.roots(x), a)))
    if i == 1:
        l34 = np.max(l34, axis=1).reshape(-1, 1)
    else:
        l34 = np.min(l34, axis=1).reshape(-1, 1)
    l34[~np.isreal(l34)] = np.nan
    theta243 = np.arccos((l34**2 + l24**2 - l23**2) / 2 / l34 / l24) * np.sign(theta432)
    theta043 = (theta024 + np.pi + theta243).reshape(-1, 1)
    p3 = p_rl(p4, l34, theta043)
    return p3, theta043


def tripleRodGroup_rrrAndrrr(revolute1, revolute4, revolute6, rod_length12, rod_length23, rod_length34, rod_length25, rod_length56, rod_length35, angle_init, i1=1, i2=1):  # 三级杆组 rrr-rrr

    def func(angle_init):
        revolute2 = p_rl(revolute1, rod_length12, angle_init.reshape(-1, 1) * (np.pi / 180), 0)
        revolute3, _ = rrr(revolute2[:, 1:3], revolute4[1:3], rod_length23, rod_length34, i1)
        revolute5, _ = rrr(revolute2[:, 1:3], revolute6[1:3], rod_length25, rod_length56, i2)
        destination = np.sqrt(np.sum((revolute3 - revolute5)**2, axis=1))
        return abs(rod_length35 - destination)

    angle_012 = root(func, angle_init)
    return angle_012.x.reshape(-1, 1) * (np.pi / 180)


def gradient(x, y, l0, d, function):  # 优化函数 梯度下降
    k = np.diff(d[:, 2:4]) / np.diff(d[:, 0:2])
    x2 = x + (l0 - y) / k
    y2 = function(x2)
    if sum(abs(l0 - y2)) < 1e-10:
        return x2, y2
    i = x < x2
    d = np.hstack([x * i + x2 * ~i, x * ~i + x2 * i, y * i + y2 * ~i, y * ~i + y2 * i])
    x, y = x2, y2
    x2, y2 = gradient(x, y, l0, d, function)
    return x2, y2


def bezierCurve(points, n=100):  # 贝塞尔曲线
    k = len(points) - 1
    t = np.linspace(0, 1, n)
    i = np.arange(k + 1).reshape(-1, 1)
    k = np.tile(np.array([k]), (len(i), 1))
    a1 = comb(k, i)
    a2 = (1 - t)**(k - i)
    a3 = t**i
    bezier = np.dot((a1 * a2 * a3).T, points)
    return bezier


def boundaryLookup(line, polygon):  # 边界查找
    polygon1 = polygon * 1
    line1 = line * 1
    i = line1[0, 1] <= line1[1, 1]
    j = i * 2 - 1
    line1 = line1[::j]
    line1_1 = line1[:-1, :]
    line1_2 = line1[1:, :]
    polygon1 = np.vstack([polygon1, line1_2])
    point1 = line1_2
    point2 = line1_1
    line2 = point2
    alpha1 = includedAngle(np.array([1, 0]), polygon1 - point2)
    point1 = point2
    point2 = polygon1[np.argmax(alpha1)]
    polygon1 = np.delete(polygon1, np.argmax(alpha1), axis=0)
    line2 = np.vstack([line2, point2])
    while np.sum(point2 == line1_2) != 2:
        alpha1 = includedAngle(point2 - point1, polygon1 - point2)
        point1 = point2
        point2 = polygon1[np.argmax(alpha1)]
        polygon1 = np.delete(polygon1, np.argmax(alpha1), axis=0)
        line2 = np.vstack([line2, point2])
    return line2[::j]


def interpolateTransform(line1, line2, total, step):  # 线1插值变换为线2,总步长,步长
    j = len(line1) > len(line2)
    (line1, line2) = (line2, line1) if j else (line1, line2)
    # 对点数少的线段插值
    k, length_lines = theta_l(np.diff(line1, axis=0))
    dx = np.diff(line1, axis=0)[:, 0]
    rate = length_lines / sum(length_lines)
    sum_points = len(line2) - 1
    num = rate * sum_points
    num = np.ceil(num)
    num[-1] = sum_points - sum(num[:-1])
    line1_1 = line1[0]
    for i in range(len(dx)):
        x = np.arange(1, num[i] + 1) * dx[i] / num[i]
        y = x * np.tan(k[i])
        line1_1 = np.vstack([line1_1, np.vstack([x, y]).T + line1[i]])

    (line1_1, line2) = (line2, line1_1) if j else (line1_1, line2)
    k = theta_l(line2 - line1_1)[0]
    dx = (line2[:, 0] - line1_1[:, 0]).reshape(-1, 1)
    dx = dx / total * step
    dy = dx * np.tan(k)
    transformation_line = np.hstack([dx, dy]) + line1_1
    return transformation_line


def steepestDescent(A, b, x):  # 最速下降解线性方程组
    r = b - A @ x  # 残差
    while np.max(np.abs(r)) > 1e-10:
        alpha = (r.T @ r) / (r.T @ A @ r)
        r = b - A @ x
        x = x + alpha * r
    return x


def conjugateGradients(A, b, x):  # 共轭梯度解线性方程组
    d = r = b - A @ x
    while np.max(np.abs(r)) > 1e-15:
        alpha = (r.T @ r) / (d.T @ A @ d)
        x = x + alpha * d
        r_next = r - alpha * A @ d
        beta = (r_next.T @ r_next) / (r.T @ r)
        r = r_next
        d = r + beta * d
    return x


# def pointsLeftProject(points, polygon, point_in=False):
#     points = np.array(points).reshape(-1, 2).astype(np.float64)
#     polygon1 = np.vstack([polygon[1:, :], polygon[0, :]])
#     in1 = ((points[:, 1:2] < polygon[:, 1]) != (points[:, 1:2] < polygon1[:, 1])) | (
#         (points[:, 1:2] > polygon[:, 1]) != (points[:, 1:2] > polygon1[:, 1]))
#     if point_in == True:
#         polygon_shapely = Polygon(polygon)
#         points_shapely = [Point(i) for i in points]
#         in1 = shapely.intersects(points_shapely, polygon_shapely).reshape(-1, 1) * in1
#     polygon2_1_x = polygon[:, 0] * in1
#     polygon2_2_x = polygon1[:, 0] * in1
#     polygon2_1_y = polygon[:, 1] * in1
#     polygon2_2_y = polygon1[:, 1] * in1
#     polygon2_1_x[~in1] = np.nan
#     polygon2_2_x[~in1] = np.nan
#     polygon2_1_y[~in1] = np.nan
#     polygon2_2_y[~in1] = np.nan
#     k21_x = (polygon2_2_x - polygon2_1_x)
#     k21_y = (polygon2_2_y - polygon2_1_y)
#     k01_y = points[:, 1:2] - polygon2_1_y
#     point_x1 = k01_y * k21_x / k21_y + polygon2_1_x
#     point_x1 = np.nan_to_num(point_x1, nan=np.inf)
#     min_point_x1 = np.nanmin(point_x1, axis=1)
#     in2 = points[:, 0] > min_point_x1
#     points[in2, 0] = min_point_x1[in2]
#     return points
