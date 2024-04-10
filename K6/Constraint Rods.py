import sys
sys.path.append(sys.path[0] + '\..')
import numpy as np
import matplotlib.pyplot as plt
from function import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class RigidRod():
    def __init__(self, x, t, mass, fixed, damping):
        self.rod_length = np.sqrt(np.sum(np.diff(x, axis=0)**2, axis=1)).reshape(-1, 1)
        self.fixed = fixed
        self.mass = mass
        self.inverse_mass = 1 / mass
        self.x = x.astype(np.float64)
        self.v = np.zeros(self.x.shape)
        self.t = t
        self.damping = damping
        self.constraint_func = 0

        constraint_num = len(self.x) - 1
        arr1 = np.zeros([constraint_num, constraint_num + 1])
        arr1_index = np.arange(constraint_num)
        arr1[arr1_index, arr1_index] = -1
        arr1[arr1_index, arr1_index + 1] = 1
        self.arr1 = np.repeat(arr1, 3).reshape(arr1.shape[0], -1)

    def forward(self):
        self.force_a = np.array([0, 0, -10])
        self.force_a = np.tile(self.force_a, (self.x.shape[0], 1))
        self.constraint()
        a = self.force_total * self.inverse_mass
        a[self.fixed, :] = 0
        self.v = self.v * self.damping + a * self.t
        self.x = self.x + self.v * self.t

    def constraint(self):
        kp = 1e4
        kd = 1e2
        inverse_mass_matrix = self.inverse_mass * np.eye(3 * len(self.x))
        vector_x = np.diff(self.x, axis=0)
        vector_v = np.diff(self.v, axis=0)
        distance_square = np.sum(vector_x**2, axis=1).reshape(-1, 1)
        constraint_func0 = self.constraint_func
        self.constraint_func = 0.5 * (distance_square - self.rod_length**2)

        arr2 = np.tile(vector_x, len(self.x))
        jacobian = arr2 * self.arr1
        arr3 = np.tile(vector_v, len(self.v))
        jacobian_d = arr3 * self.arr1

        matrix_A = jacobian @ inverse_mass_matrix @ jacobian.T
        matrix_B = -jacobian_d @ self.v.reshape(-1, 1) - jacobian @ inverse_mass_matrix @ self.force_a.reshape(-1, 1) - \
            kp * self.constraint_func - kd * (self.constraint_func - constraint_func0) / self.t
        lamuda = np.linalg.solve(matrix_A, matrix_B)
        force_constraint = (jacobian.T @ lamuda).reshape(-1, 3)
        self.force_total = force_constraint + self.force_a


rod_start = np.array([-10, 0, 0])
rod_end = np.array([10, 10, 10])
rod_node_num = 100
rod = np.vstack([np.linspace(rod_start[0], rod_end[0], rod_node_num),
                 np.linspace(rod_start[1], rod_end[1], rod_node_num),
                 np.linspace(rod_start[2], rod_end[2], rod_node_num)]).T
rod = RigidRod(x=rod, t=1e-2, mass=1, fixed=np.array([0]), damping=0.995)
length1 = stringLength(rod.x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_zlim(-30, 30)
ax.set_box_aspect([1, 1, 1])
plot_list = list(range(2))


def update(i):
    global plot_list
    try:
        [i.remove() for i in plot_list]
    except:
        pass
    for j in range(3):
        rod.forward()
    plot_list[0], = ax.plot(rod.x[:, 0], rod.x[:, 1], rod.x[:, 2], 'purple')
    plot_list[1] = ax.scatter(rod.x[:, 0], rod.x[:, 1], rod.x[:, 2], c='purple', s=1)
    print(stringLength(rod.x) - length1)


ani = FuncAnimation(fig, update, frames=20, interval=10, repeat=True)
plt.show()
