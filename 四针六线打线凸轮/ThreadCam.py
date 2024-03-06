import sys
sys.path.append('./..')
import shapely
import numpy as np
import matplotlib.pyplot as plt
from data import Cam_a
from function import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from shapely.geometry import Point, LineString, Polygon
np.set_printoptions(linewidth=200)

du = 180 / np.pi
hd = np.pi / 180


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
            if collision_state == 'in':  # 在轮廓内为碰撞
                side_collision_index[side_collision_index] = shapely.intersects(interpolate_x_points, bound_polygon)
            elif collision_state == 'out':  # 在轮廓外为碰撞
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

    def StraightCollision(self, collision_index, normal, arc_r):  # 与平面碰撞
        distance = np.sum(self.x[collision_index, 1:3] * normal, axis=-1).reshape(-1, 1)
        penetration_depth = arc_r - distance
        normal = np.insert(normal, 0, 0, axis=1)
        self.CollisionConstraint(collision_index, penetration_depth, normal)

    def ParticleInCamIndex(self, particle_positions, cam_tangency_points):  # 求点在凸轮内位置所对应的圆弧的索引
        theta = includedAngle(np.array([1, 0]), particle_positions).reshape(-1, 1)
        cam_angle_range = includedAngle(np.array([1, 0]), cam_tangency_points)
        theta_index = (theta > cam_angle_range)
        theta_below = theta_index * cam_angle_range
        nobelow_index = np.sum(theta_index, axis=-1) == 0
        theta_below[nobelow_index] = cam_angle_range
        theta_index = np.argwhere(theta_below == np.max(theta_below, axis=-1).reshape(-1, 1))[:, 1]
        return theta_index

    def Collision1(self, cam, cam_index):  # 凸轮碰撞
        # 检测节点是否发生碰撞
        cam_front_face = 1.2
        cam_back_face = -1.2
        in1 = (self.x[:, 0] > cam_back_face) * (self.x[:, 0] < cam_front_face)
        lines_points = [Point(i) for i in self.x[in1][:, [1, 2]]]
        bound_polygon = Polygon(cam.p[cam_index])
        in1[in1] = shapely.intersects(lines_points, bound_polygon)
        if np.sum(in1) > 0:
            # 侧面碰撞
            # 与凸轮前面碰撞
            self.SideCollision(cam_front_face, in1, bound_polygon, np.array([1, 0, 0]))
            # 与凸轮后面碰撞
            self.SideCollision(cam_back_face, in1, bound_polygon, np.array([-1, 0, 0]))
            if np.sum(in1) == 0:
                return

            # 圆弧面碰撞
            # 检测与节点碰撞的圆弧的序号,取出相应圆弧的圆心、半径、法向
            theta_index = self.ParticleInCamIndex(self.x[in1, 1:3], cam.tangency_points[cam_index])
            straight_index = cam.cam_info[theta_index, 3] == 0
            arc_in1 = np.copy(in1)
            straight_in1 = np.copy(in1)
            arc_in1[arc_in1] = ~straight_index
            straight_in1[straight_in1] = straight_index
            if np.sum(arc_in1) > 0:
                self.ArcCollision(arc_in1,
                                  cam.arc_c[cam_index, theta_index[~straight_index]],
                                  cam.arc_r[theta_index[~straight_index]],
                                  cam.arc_t[theta_index[~straight_index]])
            if np.sum(straight_in1) > 0:
                self.StraightCollision(straight_in1,
                                       cam.arc_c[cam_index, theta_index[straight_index]],
                                       cam.arc_r[theta_index[straight_index]])


rope_node_num = 20
rope1_start_point = np.array([2.3, -1.32867834, 17.8])
rope1_end_point = np.array([-2.3, -1.32867834, 17.8])
rope1 = np.vstack([np.linspace(rope1_start_point[0], rope1_end_point[0], rope_node_num),
                   np.linspace(rope1_start_point[1], rope1_end_point[1], rope_node_num),
                   np.linspace(rope1_start_point[2], rope1_end_point[2], rope_node_num)]).T
rope1 = Rope(rope1, spring_k=2e3, damping_k=5, m=1, fixed=[0, -1], t=2e-2)

cam1 = Cam_a()
cam1.SetCamOutline()
# cam1.Rotate(90 * hd)
cam1.Rotate(np.arange(360) * hd)

length1 = np.zeros(360)
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)
ax1.set_xlim(-10, 10)
ax1.set_ylim(-30, 30)
ax1.set_zlim(-30, 30)
ax1.set_box_aspect([20, 60, 60])
ax2.set_xlim(0, 360)
ax2.set_ylim(0, 30)
ax2.set_xticks(np.arange(0, 360, 20))
ax2.set_yticks(np.arange(0, 30, 1))
ax2.grid()

plot_dict = dict()


def update(i):
    global plot_dict
    try:
        [plot_dict[i].remove() for i in plot_dict]
    except:
        pass
    for _ in range(3):
        rope1.Forward()
        rope1.Collision1(cam1, i)

    length1[i] = stringLength(rope1.x)

    # plot_dict['particle'] = ax1.scatter(rope1.x[:, 0], rope1.x[:, 1], rope1.x[:, 2], color='y', s=10)
    plot_dict['line'], = ax1.plot(rope1.x[:, 0], rope1.x[:, 1], rope1.x[:, 2], color='purple', linewidth=1)
    plot_dict['cam1'], = ax1.plot3D(np.tile(1.2, 234), cam1.p[i, :, 0], cam1.p[i, :, 1], 'r')
    plot_dict['cam2'], = ax1.plot3D(np.tile(-1.2, 234), cam1.p[i, :, 0], cam1.p[i, :, 1], 'r')
    plot_dict['length'], = ax2.plot(np.arange(i), length1[:i], 'b')


# ax.view_init(elev=0, azim=0)
ani = FuncAnimation(fig, update, frames=360, interval=5, repeat=True)
# ani.save("animation_cam.gif", fps=25, writer="pillow")

plt.show()
