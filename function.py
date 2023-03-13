import numpy as np

def theta_l(p): #求向量与横轴正方向夹角 2维
    a = np.array([1,0])
    try:
        i = (p[:,1]<0)*-2+1
        l = np.sqrt(np.sum((p)**2,axis=1))
        theta = np.arccos(np.dot(p,a)/l)*i
    except:
        i = (p[1]<0)*-2+1
        l = np.sqrt(sum((p)**2))
        theta = np.arccos(np.dot(p,a)/l)*i
    return theta.reshape([-1,1]),l.reshape([-1,1])

def p_rl(p0,l,theta,*args): #2维;3维坐标以args[0]轴方向为旋转中心,arg[0]:x=0,y=1,z=2
    p1 = np.hstack([np.cos(theta),np.sin(theta)])
    if args==():
        return p0+l*p1
    try:
        p1 = np.insert(p1,args[0],0,axis=1)
    except:
        p1 = np.insert(p1,args[0],0)
    return p0+l*p1

def p_rp(p0,p1,theta,axis=1): #3维坐标绕过p0点且垂直于坐标轴的直线旋转,axis:x=0,y=1,z=2
    r0 = np.hstack([np.cos(theta),-np.sin(theta)])
    r0 = np.insert(r0,axis,0,axis=1)
    r1 = np.hstack([np.sin(theta),np.cos(theta)])
    r1 = np.insert(r1,axis,0,axis=1)
    p2 = np.hstack([np.sum(r0*(p1-p0),axis=1).reshape([-1,1]),
                    np.sum(r1*(p1-p0),axis=1).reshape([-1,1])])
    p0[axis] = 0
    p2 = np.insert(p2,axis,p1[axis],axis=1)+p0
    return p2

def rrr(p2,p4,l23,l34,i=1): #曲柄摇杆 2维  i:theta243正,负=1,-1
    try:
        l23 = l23.reshape([-1,1])
    except:
        pass
    theta042,l24 = theta_l(p2-p4)
    theta243 = np.arccos((l34**2+l24**2-l23**2)/2/l34/l24)*i
    theta043 = (theta042+theta243).reshape([-1,1])
    p3 = p_rl(p4,l34,theta043)
    return p3,theta043

def rrp(p5,p2,theta054,theta543,l23,l34,i=1): #曲柄滑块 2维 i:theta053==theta033_:1
    p6 = p_rl(p5,l34,theta054+theta543+np.pi)
    theta026,l26 = theta_l(p6-p2)
    theta263 = -theta026+np.pi+theta054
    a1 = 1
    a2 = -2*l26*np.cos(theta263)
    a3 = l26**2-l23**2
    a = np.concatenate((np.tile(np.array([a1]),a3.shape),a2,a3),axis=1)
    l36 = np.array(list(map(lambda x:np.roots(x),a)))
    if i==1:
        l36 = np.max(l36,axis=1).reshape([-1,1])
    else:
        l36 = np.min(l36,axis=1).reshape([-1,1])
    l36[~np.isreal(l36)]=np.nan
    p3 = p_rl(p6,l36,theta054)
    p4 = p_rl(p5,l36,theta054)
    return p3,p4

def rpr(p2,p4,l23,theta432,i=1): #l12<l14+l23有多解
    theta024,l24 = theta_l(p4-p2)
    a1 = 1
    a2 = -2*l23*np.cos(theta432)
    a3 = l23**2-l24**2
    a = np.concatenate((np.tile(np.array([a1]),a3.shape),
                        np.tile(np.array([a2]),a3.shape),
                        a3),axis=1)
    l34 = np.array(list(map(lambda x:np.roots(x),a)))
    if i==1:
        l34 = np.max(l34,axis=1).reshape([-1,1])
    else:
        l34 = np.min(l34,axis=1).reshape([-1,1])
        print(l34)
    l34[~np.isreal(l34)]=np.nan
    theta243 = np.arccos((l34**2+l24**2-l23**2)/2/l34/l24)*np.sign(theta432)
    theta043 = (theta024+np.pi+theta243).reshape([-1,1])
    p3 = p_rl(p4,l34,theta043)
    return p3,theta043

# def rrp(p5,p2,theta054,theta543,l23,l34,i=1): #曲柄滑块 2维 i:theta053==theta033_:1
#     s = np.array([np.cos(theta054),np.sin(theta054)])
#     d23 = np.cross(p2-p5,s)-l34*np.sin(theta543)
#     theta023 = (theta054-np.pi*((i+1)/2)-np.arcsin(d23/l23)*i).reshape([-1,1])
#     theta034 = theta054+theta543
#     p3 = p2+l23*np.hstack([np.cos(theta023),np.sin(theta023)])
#     p4 = p_rl(p3,l34,theta034)
#     return p3,p4

# def rrp(p5,p2,theta543,theta054,theta033_,l23,l34): #曲柄滑块 2维
#     p5 = p_rl(p5,l34,theta054+theta543+np.pi)
#     s = np.array([np.cos(theta033_),np.sin(theta033_)])
#     d23 = np.cross(p2-p5,s)
#     theta023 = (theta033_-np.arcsin(d23/l23)+np.pi).reshape([-1,1])
#     theta034 = theta054+theta543
#     p3 = p_rl(p2,l23,theta023)
#     p4 = p_rl(p3,l34,theta034)
#     return p3,p4