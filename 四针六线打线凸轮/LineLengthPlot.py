import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def inputThreadCamLineLength():
    length1 = np.zeros(360)
    with open("./ThreadCamLineLength.dat", "r") as f:
        for i in range(len(length1)):
            length1[i] = f.readline()
    return length1


def inputThreadPickLineLength():
    length2 = np.zeros(360)
    with open("./ThreadPickLineLength.dat", "r") as f:
        for i in range(len(length2)):
            length2[i] = f.readline()
    return length2


def lineLengthPlot():
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 25)
    ax1.set_xticks(np.arange(0, 360, 20))
    ax1.set_xticks(np.arange(0, 360, 10), minor=True)
    ax1.set_yticks(np.arange(0, 25, 1))
    ax1.grid()
    ax1.grid(which="minor", alpha=0.6)
    length1 = inputThreadCamLineLength()
    length1 = length1 - np.min(length1)
    length2 = inputThreadPickLineLength()
    length2 = length2 - np.min(length2)
    plot_dict = dict()

    ax1.plot([], [], c='b', label='打线凸轮')
    ax1.plot([], [], c='orange', label='挑线杆')
    ax1.legend()

    def update(i):
        nonlocal plot_dict
        try:
            [plot_dict[j].remove() for j in plot_dict]
        except:
            pass
        plot_dict['cam'], = ax1.plot(np.arange(i), length1[:i], c='b')
        plot_dict['rod'], = ax1.plot(np.arange(i), length2[:i], c='orange')

    ani = FuncAnimation(fig, update, frames=360, interval=5, repeat=True)
    # ani.save("animation_diff.gif", fps=25, writer="pillow")
    plt.show()


lineLengthPlot()
