import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__), '../..'))
import shapely
import numpy as np
import matplotlib.pyplot as plt
from k6_data import *
from function import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from shapely.geometry import Point, LineString, Polygon

k6_data(p, l, theta)


class Rope():
    def __init__(self, line, spring_k, damping_k, m, fixed, t):
        self.original_dist = stringLength(line) / len(line)
        self.spring_k = spring_k
        self.damping_k = damping_k
        self.fixed = fixed
        self.m = m
        self.x = line.astype(np.float64)
        self.v = np.zeros(self.x.shape)
        self.a = np.zeros(self.x.shape)
        self.t = t

    def Forward(self):
        force = self.ForceSpring()
        # 数值积分求解位置
        self.a = force / self.m + np.array([0, 0, -10])
        self.a[self.fixed] = 0
        self.v = self.v + self.a * self.t
        self.v = self.v * np.exp(-self.damping_k * self.t)  # 速度衰减
        self.x0 = self.x
        self.x = self.x0 + self.v * self.t

    def ForceSpring(self):  # 弹力
        x_diff1 = np.diff(self.x, axis=0)  # 相邻点坐标差
        x_diff2 = x_diff1[:-1] + x_diff1[1:]  # 中间隔一点的两点坐标差
        particle_dist1 = np.sqrt(np.sum(x_diff1**2, axis=1)).reshape(-1, 1)  # 相邻点距离
        particle_dist2 = np.sqrt(np.sum(x_diff2**2, axis=1)).reshape(-1, 1)  # 中间隔一点的两点距离
        direction1 = x_diff1 / particle_dist1
        direction2 = x_diff2 / particle_dist2
        # 结构弹簧弹力
        force_structure_spring = -self.spring_k * (particle_dist1 - self.original_dist) * direction1
        force_structure_spring = force_structure_spring - \
            (np.sum(self.v[1:] * direction1, axis=1)).reshape(-1, 1) * direction1 * self.damping_k  # 结构弹簧阻尼
        force_structure_spring = np.vstack([np.array([0, 0, 0]), force_structure_spring]) - np.vstack([force_structure_spring, np.array([0, 0, 0])])
        # 弯曲弹簧弹力
        force_bending_spring = -0.1 * self.spring_k * (particle_dist2 - 2 * self.original_dist) * direction2
        force_bending_spring = force_bending_spring - (np.sum(self.v[2:] * direction2, axis=1)).reshape(-1, 1) * direction2 * self.damping_k  # 弯曲弹簧阻尼
        force_bending_spring = np.vstack([np.array([[0, 0, 0], [0, 0, 0]]), force_bending_spring]) - \
            np.vstack([force_bending_spring, np.array([[0, 0, 0], [0, 0, 0]])])
        return force_structure_spring + force_bending_spring

    def CollisionConstraint(self, index, penetration_depth, normal):  # 碰撞约束
        self.x[index] = self.x[index] + penetration_depth * normal
        v_n = np.sum(self.v[index] * normal, axis=1).reshape(-1, 1) * normal
        v_t = self.v[index] - v_n
        self.v[index] = -0.4 * v_n + v_t

    def SideCollision(self, side, in1, bound_polygon, normal, collision_state: str = 'in'):  # 与边面碰撞
        side_collision_index = (((self.x0[:, 0] - side) * (self.x[:, 0] - side)) <= 0) * in1
        if np.sum(side_collision_index) > 0:
            vector_x = self.x[side_collision_index] - self.x0[side_collision_index]
            # 计算插值点坐标
            interpolate_x = ((side - self.x0[side_collision_index, 0]) / vector_x[:, 0]).reshape(-1, 1) * \
                vector_x[:, [1, 2]] + self.x0[side_collision_index, 1:3]
            # 在插值点是否与面碰撞
            interpolate_x_points = [Point(i) for i in interpolate_x]
            if collision_state == 'in':  # 在轮廓内产生碰撞
                side_collision_index[side_collision_index] = shapely.intersects(interpolate_x_points, bound_polygon)
            elif collision_state == 'out':  # 在轮廓外产生碰撞
                side_collision_index[side_collision_index] = ~shapely.intersects(interpolate_x_points, bound_polygon)
            # 穿透深度
            penetration_depth = np.abs(side - self.x[side_collision_index])
            self.CollisionConstraint(side_collision_index, penetration_depth, normal)
            in1[side_collision_index] = False

    def ArcCollision(self, collision_index, arc_center, arc_r, arc_t):  # 与弧面碰撞
        vector_x = self.x[collision_index, 1:3] - arc_center
        dist_to_c = np.sqrt(np.sum(vector_x**2, axis=1)).reshape(-1, 1)
        penetration_depth = np.abs(dist_to_c - arc_r)
        normal = np.insert(vector_x / dist_to_c * arc_t, 0, 0, axis=1)
        self.CollisionConstraint(collision_index, penetration_depth, normal)

    def Collision1(self, bound):  # 凸轮碰撞
        # 检测节点是否发生碰撞
        cam1_front_face = 2
        cam1_back_face = 1
        cam2_front_face = -1
        cam2_back_face = -2
        in1 = (self.x[:, 0] > cam1_back_face) * (self.x[:, 0] < cam1_front_face) + (self.x[:, 0] > cam2_back_face) * (self.x[:, 0] < -1)
        lines_points = [Point(i) for i in self.x[in1][:, [1, 2]]]
        bound_polygon = Polygon(bound)
        in1[in1] = shapely.intersects(lines_points, bound_polygon)
        if np.sum(in1) > 0:
            # 侧面碰撞
            # 与前凸轮前面碰撞
            self.SideCollision(cam1_front_face, in1, bound_polygon, np.array([1, 0, 0]))
            # 与前凸轮后面碰撞
            self.SideCollision(cam1_back_face, in1, bound_polygon, np.array([-1, 0, 0]))
            # 与后凸轮前面碰撞
            self.SideCollision(cam2_front_face, in1, bound_polygon, np.array([1, 0, 0]))
            # 与后凸轮后面碰撞
            self.SideCollision(cam2_back_face, in1, bound_polygon, np.array([-1, 0, 0]))
            if np.sum(in1) == 0:
                return
            # 圆弧面碰撞
            # 检测与节点碰撞的圆弧的序号,取出相应圆弧的圆心、半径、法向
            theta = includedAngle(bound[0], self.x[in1, 1:3]) * du + 270
            theta[theta > 360] -= 360
            theta = theta.reshape(-1, 1)
            theta_index = ((theta - cam_theta0[:, 0]) > 0) & ((theta - cam_theta0[:, 1]) < 0)
            index_3 = np.sum(theta_index, axis=1) == 0
            theta_index = np.sum(cam_theta0_index * theta_index, axis=1)
            theta_index[index_3] = 3
            cam_c = np.vstack(p['cam_info'].loc[theta_index, 'c'])
            cam_r = np.vstack(p['cam_info'].loc[theta_index, 'r'])
            cam_t = np.vstack(p['cam_info'].loc[theta_index, 't'])
            self.ArcCollision(in1, cam_c, cam_r, cam_t)

    def Collision2(self, bound):  # 中间过线槽碰撞
        in1 = (self.x[:, 0] > -0.5) * (self.x[:, 0] < 0.5)
        points = [Point(i) for i in self.x[in1][:, [1, 2]]]
        bound_polygon = Polygon(bound)
        in1[in1] = ~shapely.intersects(points, bound_polygon)
        if np.sum(in1) > 0:
            # 与过线槽前面碰撞
            self.SideCollision(0.5, in1, bound_polygon, np.array([1, 0, 0]), 'out')
            # 与过线槽后面碰撞
            self.SideCollision(-0.5, in1, bound_polygon, np.array([-1, 0, 0]), 'out')
            # 与右圆弧碰撞
            right_arc_collision_index = np.copy(in1)
            collision_index = ((self.x[in1, 1:3] - p['5_trough'][0, 1:3]) @ vector_right) > 0
            right_arc_collision_index[in1] = collision_index
            if np.sum(right_arc_collision_index) > 0:
                self.ArcCollision(right_arc_collision_index, p['5_trough_center'][0], 1.5, -1)
                in1[right_arc_collision_index] = False
            # 与左圆弧碰撞
            left_arc_collision_index = np.copy(in1)
            collision_index = ((self.x[in1, 1:3] - p['5_trough'][19, 1:3]) @ vector_left) > 0
            left_arc_collision_index[in1] = collision_index
            if np.sum(left_arc_collision_index) > 0:
                self.ArcCollision(left_arc_collision_index, p['5_trough_center'][1], 1.5, -1)
                in1[left_arc_collision_index] = False
            # 与上边线碰撞
            above_sideline_collision_index = np.copy(in1)
            collision_index = ((self.x[in1, 1:3] - p['5_trough'][9, 1:3]) @ vector_above) > 0
            above_sideline_collision_index[in1] = collision_index
            if np.sum(above_sideline_collision_index) > 0:
                penetration_depth = -np.dot((self.x[above_sideline_collision_index, 1:3] - bound[9]), normal_above).reshape(-1, 1)
                normal = np.insert(normal_above, 0, 0)
                self.CollisionConstraint(above_sideline_collision_index, penetration_depth, normal)
                in1[above_sideline_collision_index] = False
            # 与下边线碰撞
            below_sideline_collision_index = np.copy(in1)
            collision_index = ((self.x[in1, 1:3] - p['5_trough'][0, 1:3]) @ vector_below) > 0
            below_sideline_collision_index[in1] = collision_index
            if np.sum(below_sideline_collision_index) > 0:
                penetration_depth = -np.dot((self.x[below_sideline_collision_index, 1:3] - bound[19]), normal_below).reshape(-1, 1)
                normal = np.insert(normal_below, 0, 0)
                self.CollisionConstraint(below_sideline_collision_index, penetration_depth, normal)
                in1[below_sideline_collision_index] = False


trough_turn_angle = -37.00607503 * hd
translation_vector = np.array([5.7223540, 16.73727539])
p['5_trough'] = p_rp(p['5_trough'], trough_turn_angle)[0] + translation_vector
p['5_trough_center'] = p_rp(p['5_trough_center'], trough_turn_angle)[0] + translation_vector
vector_right = p['5_trough'][0] - p['5_trough'][19]
vector_left = p['5_trough'][19] - p['5_trough'][0]
vector_above = p['5_trough'][9] - p['5_trough'][0]
normal_above = -vector_above / np.sqrt(np.sum(vector_above**2))
vector_below = p['5_trough'][0] - p['5_trough'][9]
normal_below = -normal_above
p['5_trough'] = np.insert(p['5_trough'], 0, 0, axis=1)

rope_node_num = 40
rope1_start_point = np.array([-6, 15.16445824, 9])
rope1_end_point = np.array([6, 15.16445824, 9])
rope1 = np.vstack([np.linspace(rope1_start_point[0], rope1_end_point[0], rope_node_num),
                   np.linspace(rope1_start_point[1], rope1_end_point[1], rope_node_num),
                   np.linspace(rope1_start_point[2], rope1_end_point[2], rope_node_num)]).T
rope1 = Rope(rope1, spring_k=6e2, damping_k=5, m=1, fixed=[0, -1], t=2e-2)

cam_theta0 = np.vstack(p['cam_info']['theta0'])
cam_theta0_index = np.arange(12)
p['cam'] = np.insert(p['cam'], 0, 0, axis=1)
p['cam'] = p_rp(p['cam'], 90 * hd, 0)[0]
p['cam_info']['c'] = p_rp(np.vstack(p['cam_info']['c']), 90 * hd)[0].tolist()

length1 = np.zeros(360)
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)
ax1.set_xlim(-20, 20)
ax1.set_ylim(-20, 20)
ax1.set_zlim(-20, 20)
ax1.set_box_aspect([1, 1, 1])
ax2.set_xlim(0, 360)
ax2.set_ylim(0, 70)

plot_dict = dict()


def update(i):
    global plot_dict
    try:
        [plot_dict[i].remove() for i in plot_dict]
    except:
        pass
    p['cam'] = p_rp(p['cam'], 1 * hd, 0)[0]
    p['cam_info']['c'] = p_rp(np.vstack(p['cam_info']['c']), 1 * hd)[0].tolist()
    for _ in range(3):
        rope1.Forward()
        rope1.Collision1(bound=p['cam'][:, [1, 2]])
        rope1.Collision2(bound=p['5_trough'][:, [1, 2]])

    length1[i] = stringLength(rope1.x)

    # plot_dict['particle'] = ax1.scatter(rope1.x[:, 0], rope1.x[:, 1], rope1.x[:, 2], color='y', s=10)
    plot_dict['line'], = ax1.plot(rope1.x[:, 0], rope1.x[:, 1], rope1.x[:, 2], color='purple', linewidth=1)
    plot_dict['trough1'], = ax1.plot3D(np.hstack([p['5_trough'][:, 0], p['5_trough'][0, 0]]) - 0.5,
                                       np.hstack([p['5_trough'][:, 1], p['5_trough'][0, 1]]),
                                       np.hstack([p['5_trough'][:, 2], p['5_trough'][0, 2]]), 'c')
    plot_dict['trough2'], = ax1.plot3D(np.hstack([p['5_trough'][:, 0], p['5_trough'][0, 0]]) + 0.5,
                                       np.hstack([p['5_trough'][:, 1], p['5_trough'][0, 1]]),
                                       np.hstack([p['5_trough'][:, 2], p['5_trough'][0, 2]]), 'c')
    plot_dict['cam1'], = ax1.plot3D(p['cam'][:, 0] + 2, p['cam'][:, 1], p['cam'][:, 2], 'r')
    plot_dict['cam2'], = ax1.plot3D(p['cam'][:, 0] - 2, p['cam'][:, 1], p['cam'][:, 2], 'r')
    plot_dict['length'], = ax2.plot(np.arange(i), length1[:i], 'b')


# ax.view_init(elev=0, azim=0)
ani = FuncAnimation(fig, update, frames=360, interval=5, repeat=False)
# ani.save("animation_cam.gif", fps=25, writer="pillow")

plt.show()
