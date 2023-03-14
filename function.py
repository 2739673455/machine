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

def rrp(p2,p5,l23,l34,theta054,theta543,i=1): #曲柄滑块 2维 i:33_方向
    p6 = p_rl(p5,l34,theta054+theta543+np.pi)
    theta026,l26 = theta_l(p6-p2)
    a1 = 1
    a2 = -2*l26*np.cos(theta026-theta054)
    a3 = l26**2-l23**2
    a = np.concatenate((np.tile(np.array([a1]),a3.shape),a2,a3),axis=1)
    l36 = -np.array(list(map(lambda x:np.roots(x),a)))
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
    l34[~np.isreal(l34)]=np.nan
    theta243 = np.arccos((l34**2+l24**2-l23**2)/2/l34/l24)*np.sign(theta432)
    theta043 = (theta024+np.pi+theta243).reshape([-1,1])
    p3 = p_rl(p4,l34,theta043)
    return p3,theta043

du = 180/np.pi
hd = np.pi/180

pm = np.array([67.38779579,-14.50871791])
pn = np.array([203.46200448,-7.56520072])
pl = np.array([48.45361005,120.54683341])
lmq = 97.26542294
lqp = 3.57
lps = 113.61007878
lsl = 125.4
thetanpq = 90*hd

l0 = 113.50900801

x = np.linspace(0,np.pi/9,21).reshape(-1,1)
def func1(x):
    pq = p_rl(pm,lmq,x)
    pp,thetanp = rpr(pq,pn,lqp,thetanpq)
    ps,thetals = rrr(pp,pl,lps,lsl,-1)
    thetaqs,lqs = theta_l(ps-pq)
    return lqs
y = func1(x)
#0
index0 = np.argwhere(np.abs(y-l0)==np.min(np.abs(y-l0)))[0,0]
x0 = x[index0]
y0 = y[index0]
sign_d0 = np.sign(l0-y0)
sign_dy = np.sign(np.diff(y,axis=0)[index0])
index1 = index0+int(sign_d0*sign_dy)*1
x1 = x[index1]
y1 = y[index1]
x = np.hstack([x0,x1])
y = np.hstack([y0,y1])
print(x*du,y)
def approximation(x,y,l0):
    k = np.diff(y)/np.diff(x)
    x2 = (l0-y[0])/k+x[0]
    y2 = func1(x2)
    sign_d0 = np.sign(l0-y2)
    sign_dy = np.sign(np.diff(y))
    if abs(l0-y2)<1e-10:
        return x2,y2
    elif sign_d0*sign_dy>0:
        x[0] = x2
        y[0] = y2 
    else:
        x[1] = x2
        y[1] = y2
    print(x*du,y)
    x2,y2 = approximation(x,y,l0)
    return x2,y2
x2,y2 = approximation(x,y,l0)
print(x2*du)