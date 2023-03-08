import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def crank_rocker(p1,p2,p4,l23,l34,clockwise=1): #曲柄摇杆
    a=[1,0,1]
    l14=np.sqrt(sum((p1-p4)**2))
    l24=np.sqrt(np.sum((p2-p4)**2,axis=1))
    theta041=np.arccos((p1[0]-p4[0])/l14)*(p1[1]-p4[1])/abs(p1[1]-p4[1])
    theta241=np.arcsin(np.cross(p2-p4,p1-p4)/l14/l24)
    theta342=np.arccos((l34**2+l24**2-l23**2)/2/l34/l24)*clockwise
    theta043=(theta041-theta342-theta241-2*np.pi*a[clockwise]).reshape(-1,1)
    p3=p4+l34*np.hstack((np.cos(theta043),np.sin(theta043)))
    return p3,theta043

def crank_slider(p0,p2,theta0,l23,l34,axis=1): #曲柄滑块
    s = np.array([np.cos(theta0),np.sin(theta0)])
    p2_ = np.delete(p2,axis,axis=1)
    p0_ = np.delete(p0,axis)
    d = np.cross(p2_-p0_,s)-l34
    theta023 = theta0-np.pi-np.arcsin(d/l23).reshape([-1,1])
    p3 = p2_+l23*np.hstack([np.cos(theta023),np.sin(theta023)])
    p3 =np.insert(p3,axis,p0[axis],axis=1)
    return p3

def p_rp(p0,p1,theta,axis=1): #三维坐标绕过p0点且垂直于坐标轴的直线旋转,axis:x=0,y=1,z=2
    r0 = np.hstack([np.cos(theta),-np.sin(theta)])
    r0 = np.insert(r0,axis,0,axis=1)
    r1 = np.hstack([np.sin(theta),np.cos(theta)])
    r1 = np.insert(r1,axis,0,axis=1)
    p2 = np.hstack([np.sum(r0*(p1-p0),axis=1).reshape([-1,1]),
                    np.sum(r1*(p1-p0),axis=1).reshape([-1,1])])
    p0[axis] = 0
    p2 = np.insert(p2,axis,p1[axis],axis=1)+p0
    return p2

def p_rl(p0,l,theta,axis=1):
    p1 = np.hstack([np.cos(theta),np.sin(theta)])
    try:
        p1 = np.insert(p1,axis,0,axis=1)
    except:
        p1 = np.insert(p1,axis,0)
    p1 = p0+l*p1
    return p1

du = 180/np.pi
hd = np.pi/180
n = np.linspace(0,2*np.pi,361).reshape(-1,1)

p1_a = np.array([48.45361005,-15.05421454,122.99683341])
p1_e = np.array([33.34691402,-8.17324548,122.99683341])
p1_g2 = np.array([0,0,0])
p1_h2 = np.array([68,0,-36])
p1_h2_1 = np.array([68,5,-36])
p1_h2_2 = np.array([68,-4.99,-36])
p1_j = np.array([68,5,-15.4])
p1_n = np.array([203,0,-7])

l1_ab = 2.45
l1_bc1 = 125.4 

l1_c8d = 17.15
l1_ef = 2.05
l1_fg1 = 123.4
l1_g1g2 = 29.2
l1_g3g4 = 50
l1_g4h1 = 57.1
l1_h1h2 = 38.73512618
l1_h2h3 = 3.97
l1_h2h5 = 21.5
l1_h5c3 = 97.26542294
l1_i1i2 = 22.1
l1_i2i3 = 71
l1_jk1 = 12.6
l1_h2k2 = 23.7
l1_k2i2 = 25

l1_l8m = 16.1
l1_no=0.73

theta_shangzhou = 24.48885351*hd

theta1_g2g3g4 =-40.32593142*hd
theta1_h1h2h3 = 258.08086954*hd
theta1_h2h3i1 = -90*hd
theta1_k1h2k2 = 275*hd

theta1_0jk1 = 80*hd
theta1_0no = -45*hd

theta1_0ab = n
theta1_0ef = theta1_0ab-95*hd

p1_b=p1_a+l1_ab*np.hstack((np.cos(theta1_0ab)*np.sin(theta_shangzhou),
                           np.cos(theta1_0ab)*np.cos(theta_shangzhou),
                           np.sin(theta1_0ab)))
p1_f=p1_e+l1_ef*np.hstack((np.cos(theta1_0ef)*np.sin(theta_shangzhou),
                           np.cos(theta1_0ef)*np.cos(theta_shangzhou),
                           np.sin(theta1_0ef)))
l1_ab_xz = np.sqrt(l1_ab**2-(p1_b[:,1]-p1_a[1])**2)
l1_bc1_xz = np.sqrt(l1_bc1**2-p1_b[:,1]**2)

l1_fg1_xz = np.sqrt(l1_fg1**2-p1_f[:,1]**2)
p1_g1,theta1_0g2g1 = crank_rocker(p1_e[[0,2]],p1_f[:,[0,2]],p1_g2[[0,2]],l1_fg1_xz,l1_g1g2)
p1_g1 = np.insert(p1_g1,1,p1_g2[1],axis=1)
p1_g4 = p_rl(np.array([48.11877609,0,-0.35674442]),l1_g3g4,np.pi+theta1_g2g3g4,axis=1)
p1_g4 = p_rp(p1_g2,p1_g4,theta1_0g2g1,axis=1)

p1_h1,theta1_0h2h1 = crank_rocker(p1_g2[[0,2]],p1_g4[:,[0,2]],p1_h2[[0,2]],l1_g4h1,l1_h1h2)
p1_h1 = np.insert(p1_h1,1,p1_g2[1],axis=1)
theta1_0h2h3 = theta1_0h2h1+theta1_h1h2h3
p1_h3 = p_rp(p1_h2,np.array([3.97,0,0])+p1_h2,theta1_0h2h3,axis=1)
p1_h5 = p_rp(p1_h2,np.array([0,-4.99,21.5])+p1_h2,theta1_0h2h3,axis=1)

p1_k1 = p_rl(p1_j,l1_jk1,theta1_0jk1,axis=1)
l1_k1h2 = np.sqrt((p1_k1[2]-p1_h2[2])**2+(p1_k1[0]-p1_h2[0])**2)
theta1_0h2k2 = np.arccos((p1_k1[0]-p1_h2[0])/l1_k1h2)+theta1_k1h2k2
p1_k2 = p_rl(p1_h2_1,l1_h2k2,theta1_0h2k2,axis=1)
p1_o = p_rl(p1_n,l1_no,theta1_0no,axis=1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim(-10,310)
ax.set_ylim(-50,50)
ax.set_zlim(-50,150)
ax.set_box_aspect([320,100,200])
line=dict()
def update(i):
    global line
    try:
        line[1.1].remove()
        line[1.2].remove()
        line[1.3].remove()
        line[1.4].remove()
        line[1.5].remove()
    except:
        pass
    x1_1=[p1_e[0],p1_f[i,0],p1_g1[i,0],p1_g2[0],p1_g4[i,0],p1_h1[i,0],p1_h2[0],p1_h3[i,0]]
    y1_1=[p1_e[1],p1_f[i,1],p1_g1[i,1],p1_g2[1],p1_g4[i,1],p1_h1[i,1],p1_h2[1],p1_h3[i,1]]
    z1_1=[p1_e[2],p1_f[i,2],p1_g1[i,2],p1_g2[2],p1_g4[i,2],p1_h1[i,2],p1_h2[2],p1_h3[i,2]]
    line[1.1],=ax.plot3D(x1_1,y1_1,z1_1,'r')
    x1_2=[p1_h2[0],p1_h2_2[0],p1_h5[i,0]]
    y1_2=[p1_h2[1],p1_h2_2[1],p1_h5[i,1]]
    z1_2=[p1_h2[2],p1_h2_2[2],p1_h5[i,2]]
    line[1.2],=ax.plot3D(x1_2,y1_2,z1_2,'b')
    x1_3=[p1_j[0],p1_k1[0],p1_h2_1[0],p1_k2[0]]
    y1_3=[p1_j[1],p1_k1[1],p1_h2_1[1],p1_k2[1]]
    z1_3=[p1_j[2],p1_k1[2],p1_h2_1[2],p1_k2[2]]
    line[1.3],=ax.plot3D(x1_3,y1_3,z1_3,'g')
    x1_4=[p1_n[0],p1_o[0]]
    y1_4=[p1_n[1],p1_o[1]]
    z1_4=[p1_n[2],p1_o[2]]
    line[1.4],=ax.plot3D(x1_4,y1_4,z1_4,'g')
    x1_5=[p1_a[0],p1_b[i,0]]
    y1_5=[p1_a[1],p1_b[i,1]]
    z1_5=[p1_a[2],p1_b[i,2]]
    line[1.5],=ax.plot3D(x1_5,y1_5,z1_5,'b')

ani=FuncAnimation(fig,update,frames=range(0,len(n),1),interval=1,repeat=True)
# ani.save("animation.gif", fps=25, writer="pillow")

plt.show()