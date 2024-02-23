import numpy as np
from function import *

du = 180/np.pi
hd = np.pi/180
n = np.linspace(0,359*hd,360).reshape(-1,1)
ONE = np.ones([len(n),1])

def solve(p,l,theta,n):

    # # 针杆
    # theta['1_ab'] = theta['1_0ab']+n
    # p['1_b']  = p_rl(p['1_a'],l['1_ab'],theta['1_ab'],0)
    # p['1_c']  = np.hstack([ONE*p['1_a'][0],ONE*p['1_a'][1],(p['1_b'][:,2]-np.sqrt(l['1_bc']**2-p['1_b'][:,1]**2)).reshape(-1,1)])
    # p['1_d']  = p['1_c']  - np.array([0,0,l['1_cd']])
    # p['1_d1'] = p['1_d']  + np.array([l['1_dd1'],0,0])
    # p['1_d3'] = p['1_d']  - np.array([l['1_dd1'],0,0])
    # p['1_e2'] = p['1_d']  + np.array([0,0,-l['1_e2d']])
    # p['1_e1'] = p['1_e2'] + np.array([l['1_dd1'],0,-l['1_e1e2']])
    # p['1_e3'] = p['1_e2'] - np.array([l['1_dd1'],0,-l['1_e1e2']])
    # p['1_f2'] = p['1_d']  + np.array([0,0,-l['1_f2d']])
    # p['1_f1'] = p['1_f2'] + np.array([l['1_dd1'],0,-l['1_e1e2']])
    # p['1_f3'] = p['1_f2'] - np.array([l['1_dd1'],0,-l['1_e1e2']])
    # p['1_g2'] = p['1_d']  + p['1_g2']
    # p['1_g1'] = p['1_g2'] + np.array([l['1_dd1'],0,0])
    # p['1_g3'] = p['1_g2'] - np.array([l['1_dd1'],0,0])

    # 挑线
    theta['2_0ab'] = theta['1_0ab']+theta['2_12']+n
    p['2_b'] = p_rl(p['2_a'],l['2_ab'],theta['2_0ab'],0)
    p['2_c'] = np.hstack([ONE*0,ONE*p['2_a'][1],(p['2_b'][:,2]-np.sqrt(l['2_bc']**2-p['2_b'][:,1]**2)).reshape(-1,1)])
    p['2_c'][:,0] = p['2_d'][0]+np.sqrt(l['2_cd']**2-(p['2_c'][:,2]-p['2_d'][2])**2)
    p['2_b'][:,0] = p['2_c'][:,0]
    p['2_a'] = np.tile(p['2_a'],[len(n),1])
    p['2_a'] = p['2_a'].astype(np.float64)
    p['2_a'][:,0] = p['2_c'][:,0]
    theta['2_0dc'] = theta_l(p['2_c'][:,[0,2]]-p['2_d'][[0,2]])[0]
    theta['2_0ex'] = theta['2_0dc']+theta['2_cex']
    p['2_f1'] = p_rp(p['2_f1'],theta['2_0ex'],1)+p['2_e']
    p['2_f2'] = p['2_f1']+np.array([0,l['2_f1f2'],0])
    p['2_f3'] = p['2_f1']+np.array([0,l['2_f1f3'],0])
    p['2_g1'] = p_rp(p['2_g1'],theta['2_0ex'],1)+p['2_e']
    p['2_g2'] = p['2_g1']+np.array([0,l['2_f1f2'],0])
    p['2_g3'] = p['2_g1']+np.array([0,l['2_f1f3'],0])
    p['2_h']  = p_rp(p['2_h'],theta['2_0ex'],1)+p['2_e']

    index_l = ['2_k1','2_j1','2_f1','2_g1','2_i1']
    index_m = ['2_k2','2_j2','2_f2','2_g2','2_i2']
    index_r = ['2_k3','2_j3','2_f3','2_g3','2_i3']
    l['length_l'] = np.zeros(n.shape)
    l['length_m'] = np.zeros(n.shape)
    l['length_r'] = np.zeros(n.shape)
    for i in range(1,len(index_l)):
        l['length_l']+=length(p[index_l[i-1]],p[index_l[i]])
        l['length_m']+=length(p[index_m[i-1]],p[index_m[i]])
        l['length_r']+=length(p[index_r[i-1]],p[index_r[i]])
    l['length_l'] = l['length_l']-l['length_l'][0]+100
    l['length_m'] = l['length_m']-l['length_m'][0]+50
    l['length_r'] = l['length_r']-l['length_r'][0]

def length(p1,p2):
    try:
        return (np.sqrt(np.sum((p2-p1)**2,axis=1))).reshape(-1,1)
    except:
        return np.sqrt(np.sum((p2-p1)**2))

if __name__=='__main__':
    from k6_data import *
    from k6_plot import *
    k6_data(p,l,theta)
    solve(p,l,theta,n)
    k6_plot(p,l,theta,n)