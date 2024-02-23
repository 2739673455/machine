import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation

def k6_plot(self,p,l,theta,n,drawing_times):
    du = 180/np.pi
    self.ax1.cla()
    self.ax1.set_xlim(-160,10)
    self.ax1.set_ylim(-10,80)
    self.ax1.set_zlim(-60,60)
    self.ax1.set_box_aspect([170,90,120])
    self.ax1.view_init(elev=30, azim=30)
    x = [p['2_i1'][0],p['2_i2'][0],p['2_i3'][0],p['2_j1'][0],p['2_j2'][0],p['2_j3'][0],p['2_k1'][0],p['2_k2'][0],p['2_k3'][0]]
    y = [p['2_i1'][1],p['2_i2'][1],p['2_i3'][1],p['2_j1'][1],p['2_j2'][1],p['2_j3'][1],p['2_k1'][1],p['2_k2'][1],p['2_k3'][1]]
    z = [p['2_i1'][2],p['2_i2'][2],p['2_i3'][2],p['2_j1'][2],p['2_j2'][2],p['2_j3'][2],p['2_k1'][2],p['2_k2'][2],p['2_k3'][2]]
    self.ax1.plot3D(x,y,z,'ro',markersize=3)

    color_list = ['k', 'r', 'y', 'g', 'b']
    order = drawing_times%5
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    self.ax2.plot(n*du, l['length_l'], lw=1, ls='-',  color=color_list[order], label=str(order))
    self.ax2.plot(n*du, l['length_m'], lw=1, ls='--', color=color_list[order])
    self.ax2.plot(n*du, l['length_r'], lw=1, ls=':',  color=color_list[order])
    self.ax2.legend()
    self.ax2.grid(True)
    self.ax2.set_title('——  左       - -  中       :  右')

    plot_mechanism, = self.ax1.plot3D([],[],[],'-',color='slateblue')
    plot_rope1, = self.ax1.plot3D([],[],[],'r-',lw=1)
    plot_rope2, = self.ax1.plot3D([],[],[],'r--',lw=1)
    plot_rope3, = self.ax1.plot3D([],[],[],'r:',lw=1)
    plot_points, = self.ax1.plot3D([],[],[],'ro',markersize=3)

    text_plot = dict()
    index_mechanism = ['a','b','c','d','e','k1','j1','f1','g1','i1']
    for index in index_mechanism:
        text_plot[index] = self.ax1.text(0,0,0,index)

    def update(i):
        [text_plot[i].remove() for i in text_plot]
        x = [p['2_a'][i,0],p['2_b'][i,0],p['2_c'][i,0],p['2_d'][0],p['2_e'][0]]
        y = [p['2_a'][i,1],p['2_b'][i,1],p['2_c'][i,1],p['2_d'][1],p['2_e'][1]]
        z = [p['2_a'][i,2],p['2_b'][i,2],p['2_c'][i,2],p['2_d'][2],p['2_e'][2]]
        plot_mechanism .set_data_3d(x,y,z)
        x = [p['2_k1'][0],p['2_j1'][0],p['2_f1'][i,0],p['2_g1'][i,0],p['2_i1'][0]]
        y = [p['2_k1'][1],p['2_j1'][1],p['2_f1'][i,1],p['2_g1'][i,1],p['2_i1'][1]]
        z = [p['2_k1'][2],p['2_j1'][2],p['2_f1'][i,2],p['2_g1'][i,2],p['2_i1'][2]]
        plot_rope1.set_data_3d(x,y,z)
        x = [p['2_k2'][0],p['2_j2'][0],p['2_f2'][i,0],p['2_g2'][i,0],p['2_i2'][0]]
        y = [p['2_k2'][1],p['2_j2'][1],p['2_f2'][i,1],p['2_g2'][i,1],p['2_i2'][1]]
        z = [p['2_k2'][2],p['2_j2'][2],p['2_f2'][i,2],p['2_g2'][i,2],p['2_i2'][2]]
        plot_rope2.set_data_3d(x,y,z)
        x = [p['2_k3'][0],p['2_j3'][0],p['2_f3'][i,0],p['2_g3'][i,0],p['2_i3'][0]]
        y = [p['2_k3'][1],p['2_j3'][1],p['2_f3'][i,1],p['2_g3'][i,1],p['2_i3'][1]]
        z = [p['2_k3'][2],p['2_j3'][2],p['2_f3'][i,2],p['2_g3'][i,2],p['2_i3'][2]]
        plot_rope3.set_data_3d(x,y,z)
        x = [p['2_f1'][i,0],p['2_f2'][i,0],p['2_f3'][i,0],p['2_g1'][i,0],p['2_g2'][i,0],p['2_g3'][i,0]]
        y = [p['2_f1'][i,1],p['2_f2'][i,1],p['2_f3'][i,1],p['2_g1'][i,1],p['2_g2'][i,1],p['2_g3'][i,1]]
        z = [p['2_f1'][i,2],p['2_f2'][i,2],p['2_f3'][i,2],p['2_g1'][i,2],p['2_g2'][i,2],p['2_g3'][i,2]]
        plot_points.set_data_3d(x,y,z)

        for index in index_mechanism:
            try:
                text_plot[index] = self.ax1.text(p[f'2_{index}'][i,0],p[f'2_{index}'][i,1],p[f'2_{index}'][i,2],index)
            except:
                text_plot[index] = self.ax1.text(p[f'2_{index}'][0],p[f'2_{index}'][1],p[f'2_{index}'][2],index)

    ani = FuncAnimation(self.fig,update,frames=range(0,len(n),2),interval=10,repeat=True)
#    update(0)
    self.canvas.draw()