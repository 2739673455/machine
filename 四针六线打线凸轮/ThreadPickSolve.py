import sys
sys.path.append("./..")
import numpy as np
import matplotlib.pyplot as plt
from data import *
from function import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

rod1 = ThreadPickRod()
rod1.Solve()
rope_node1 = np.array([14.5, 23.85, 9.6])
rope_node4 = np.array([-2.5, 23.85, 9.6])

distance = np.sum((rope_node1 - rod1.below_side1[:, 0]) * rod1.normal, axis=1).reshape(-1, 1)
vector = distance * rod1.normal
intersects_position = vector + rod1.below_side1[:, 0]

rope_node2 = np.copy(intersects_position)
rope_node2[:, 0:1] = (distance - 23.5) / (31.00577422 - 23.5) * 3.5 + 1.5
rope_node2[:, 0:1][rope_node2[:, 0:1] < 1.5] = 1.5

rope_node3 = np.copy(intersects_position)
rope_node3[:, 0:1] = (distance - 23.83254199) / (31.33831622 - 23.83254199) * 3.5
rope_node3[:, 0:1][rope_node3[:, 0:1] < 0] = 0

length1 = np.zeros(360)
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)
ax1.set_xlim(-5, 15)
ax1.set_ylim(-10, 40)
ax1.set_zlim(-10, 20)
ax1.set_box_aspect([20, 50, 30])
ax2.set_xlim(0, 360)
ax2.set_ylim(15, 40)
ax2.set_xticks(np.arange(0, 360, 20))
ax2.set_yticks(np.arange(15, 40, 1))
ax2.grid()

plot_dict = dict()


def update(i):
    global plot_dict
    try:
        [plot_dict[i].remove() for i in plot_dict]
    except:
        pass

    rope_node_list = np.vstack([rope_node1, rope_node2[i], rope_node3[i], rope_node4])
    length1[i] = stringLength(rope_node_list)

    # plot_dict['particle'] = ax1.scatter(rope_node_list[:, 0], rope_node_list[:, 1], rope_node_list[:, 2], color='y', s=10)
    plot_dict['line'], = ax1.plot(rope_node_list[:, 0], rope_node_list[:, 1], rope_node_list[:, 2], color='purple', linewidth=1)
    plot_dict['rod1'], = ax1.plot(rod1.below_side1[i, :, 0], rod1.below_side1[i, :, 1], rod1.below_side1[i, :, 2], 'c')
    plot_dict['rod2'], = ax1.plot(rod1.below_side2[i, :, 0], rod1.below_side2[i, :, 1], rod1.below_side2[i, :, 2], 'c')
    plot_dict['length'], = ax2.plot(np.arange(i), length1[:i], 'b')


# ax.view_init(elev=0, azim=0)
ani = FuncAnimation(fig, update, frames=360, interval=5, repeat=True)
# ani.save("animation_cam.gif", fps=25, writer="pillow")

plt.show()
