import numpy as np
import matplotlib.pyplot as plt


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
    ax1.set_yticks(np.arange(0, 25, 1))
    ax1.grid()
    length1 = inputThreadCamLineLength()
    length1 = length1 - np.min(length1)
    length2 = inputThreadPickLineLength()
    length2 = length2 - np.min(length2)
    ax1.plot(length1, label="打线凸轮")
    ax1.plot(length2, label="挑线杆")
    ax1.legend()
    plt.show()


lineLengthPlot()
