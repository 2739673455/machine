import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

def crank_rocker(p1,p2,p4,l23,l34,clockwise=1):
    a=[0,0,1]
    l14=np.sqrt(sum((p1-p4)**2))
    l24=np.sqrt(np.sum((p2-p4)**2,axis=1))
    theta041=np.arccos((p1[0]-p4[0])/l14)*(p1[1]-p4[1])/abs(p1[1]-p4[1])
    theta241=np.arcsin(np.cross(p2-p4,p1-p4)/l14/l24)
    theta342=np.arccos((l34**2+l24**2-l23**2)/2/l34/l24)*clockwise
    theta043=(theta041-theta342-theta241-2*np.pi*a[clockwise]).reshape(-1,1)
    p3=p4+l34*np.hstack((np.cos(theta043),np.sin(theta043)))
    return p3

n=np.linspace(0,2*np.pi,361).reshape(-1,1)
du=180/np.pi
hd=np.pi/180
l12=5
l23=15
l34=15
p1=np.array([0,0])
p4=np.array([-10,-10])
p2=p1+l12*np.hstack((np.cos(n),np.sin(n)))
p3=crank_rocker(p1,p2,p4,l23,l34,-1)
print(p3)


fig,ax=plt.subplots()
plt.xlim(-15,10)
plt.ylim(-20,5)
plt.axis('equal')
def update(n):
    global line
    try:
        line.remove()
    except:
        pass
    x=[p1[0],p2[n,0],p3[n,0],p4[0]]
    y=[p1[1],p2[n,1],p3[n,1],p4[1]]
    line,=ax.plot(x,y,'r')
    
ani=FuncAnimation(fig,update,frames=range(len(n)),interval=1,repeat=True)
plt.show()