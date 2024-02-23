import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d

def k6_plot(p,l,theta,n):
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1,projection="3d")
    ax1 = fig.add_subplot(1,2,2)
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-185,-170)
    ax.set_box_aspect([20,20,15])
    ax.view_init(elev=3, azim=3)
    line = dict()
    line_indexs = [1.0, 1.1, 1.2, 1.3,
                   3.0, 3.1, 3.2, 3.3,
                   0.1, 
                   'length']
    def update(i):
        nonlocal line
        try:
            [line[line_index].remove() for line_index in line_indexs]
        except:
            pass
        x = [0,0]
        y = [-20,20]
        z = [l['tph'],l['tph']]
        line[1.0], = ax.plot3D(x,y,z,'purple')
        x = [p['1_d'][i,0],p['1_f2'][i,0]]
        y = [p['1_d'][i,1],p['1_f2'][i,1]]
        z = [p['1_d'][i,2],p['1_f2'][i,2]]
        line[1.1], = ax.plot3D(x,y,z,'purple')
        x = [p['1_f1'][i,0],p['1_d1'][i,0],p['1_d3'][i,0],p['1_f3'][i,0]]
        y = [p['1_f1'][i,1],p['1_d1'][i,1],p['1_d3'][i,1],p['1_f3'][i,1]]
        z = [p['1_f1'][i,2],p['1_d1'][i,2],p['1_d3'][i,2],p['1_f3'][i,2]]
        line[1.2], = ax.plot3D(x,y,z,'purple')
        x = [p['1_e1'][i,0],p['1_e2'][i,0],p['1_e3'][i,0],p['1_g1'][i,0],p['1_g2'][i,0],p['1_g3'][i,0]]
        y = [p['1_e1'][i,1],p['1_e2'][i,1],p['1_e3'][i,1],p['1_g1'][i,1],p['1_g2'][i,1],p['1_g3'][i,1]]
        z = [p['1_e1'][i,2],p['1_e2'][i,2],p['1_e3'][i,2],p['1_g1'][i,2],p['1_g2'][i,2],p['1_g3'][i,2]]
        line[1.3] = ax.scatter3D(x,y,z,c='purple')

        x = np.hstack([p['3_wanzhen_1'][i,:,0],p['3_wanzhen_1'][i,0,0]])
        y = np.hstack([p['3_wanzhen_1'][i,:,1],p['3_wanzhen_1'][i,0,1]])
        z = np.hstack([p['3_wanzhen_1'][i,:,2],p['3_wanzhen_1'][i,0,2]])
        line[3.0], = ax.plot3D(x,y,z,c='purple')
        x = np.hstack([p['3_wanzhen_2'][i,:,0],p['3_wanzhen_2'][i,0,0]])
        y = np.hstack([p['3_wanzhen_2'][i,:,1],p['3_wanzhen_2'][i,0,1]])
        z = np.hstack([p['3_wanzhen_2'][i,:,2],p['3_wanzhen_2'][i,0,2]])
        line[3.1], = ax.plot3D(x,y,z,c='purple')
        x = np.hstack([p['3_wanzhen_1'][i,0,0],p['3_wanzhen_2'][i,0,0],p['3_k'][i,0],p['3_wanzhen_2'][i,1,0],p['3_wanzhen_1'][i,1,0],p['3_wanzhen_1'][i,2,0],p['3_k'][i,0],p['3_wanzhen_1'][i,3,0],p['3_wanzhen_1'][i,4,0],p['3_wanzhen_2'][i,4,0],p['3_k'][i,0]])
        y = np.hstack([p['3_wanzhen_1'][i,0,1],p['3_wanzhen_2'][i,0,1],p['3_k'][i,1],p['3_wanzhen_2'][i,1,1],p['3_wanzhen_1'][i,1,1],p['3_wanzhen_1'][i,2,1],p['3_k'][i,1],p['3_wanzhen_1'][i,3,1],p['3_wanzhen_1'][i,4,1],p['3_wanzhen_2'][i,4,1],p['3_k'][i,1]])
        z = np.hstack([p['3_wanzhen_1'][i,0,2],p['3_wanzhen_2'][i,0,2],p['3_k'][i,2],p['3_wanzhen_2'][i,1,2],p['3_wanzhen_1'][i,1,2],p['3_wanzhen_1'][i,2,2],p['3_k'][i,2],p['3_wanzhen_1'][i,3,2],p['3_wanzhen_1'][i,4,2],p['3_wanzhen_2'][i,4,2],p['3_k'][i,2]])
        line[3.2], = ax.plot3D(x,y,z,c='purple')
        x = [p['3_j'][i,0]]
        y = [p['3_j'][i,1]]
        z = [p['3_j'][i,2]]
        line[3.3] = ax.scatter3D(x,y,z,c='purple')

        line[0.1], = ax.plot3D(p['ring'][str(i)][:,0],p['ring'][str(i)][:,1],p['ring'][str(i)][:,2],'y')
        line['length'], = ax1.plot(np.arange(i),l['length1'][:i],'b')



    ani = FuncAnimation(fig,update,frames=range(0,len(n),1),interval=1,repeat=True)
    # ani.save("animation_wanzhen.gif", fps=25, writer="pillow")
    # update(271)

    plt.show()