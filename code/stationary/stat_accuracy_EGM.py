"""
This script compares the accuracy of the discrete-time brute force and
endogenous grid methods in a stationary setting.

accuracy_tables produces three tables:
    * comparison of BF and "true" solution on various grids;
    * comparison of EGM and "true" solution on various grids;
    * comparison of BF and EGM solution on various grids.

"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import time, classes, parameters
import matplotlib.pyplot as plt
from true_stat import true_stat_load

c1,c2 = parameters.c1, parameters.c2
colorFader = parameters.colorFader

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mu, sigma = parameters.mu, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 1, 1, 1
N_true, N_c = parameters.N_true, parameters.N_c
n_round_acc = parameters.n_round_acc
n_round_time = parameters.n_round_time
CT_dt_true = parameters.CT_dt_true
CT_dt_mid = parameters.CT_dt_mid
CT_dt_big = parameters.CT_dt_big
DT_dt = parameters.DT_dt

cols = ['$||c - c_{\textnormal{true}}||_1$',
'$||c - c_{\textnormal{true}}||_{\infty}$',
'$||V - V_{\textnormal{true}}||_1$',
'$||V - V_{\textnormal{true}}||_{\infty}$']

cols_compare = ['$||c_{E} - c_{B}||_1$',
'$||c_{E} - c_{B}||_{\infty}$',
'$||V_{E} - V_{B}||_1$',
'$||V_{E} - V_{B}||_{\infty}$']

"""
Functions for data construction and table creation
"""

def true_DT_stat(DT_dt):
    true_val = {}
    destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('DT',int(10**3*DT_dt))
    if os.path.exists(destin):
        V = pd.read_csv(destin)
    destin = '../../main/output/true_c_{0}_stat_{1}.csv'.format('DT',int(10**3*DT_dt))
    if os.path.exists(destin):
        c = pd.read_csv(destin)
    true_val['DT'] = np.array(V), np.array(c)
    return true_val

def accuracy_data(true_val,N_set,DT_dt):
    """
    Pre-allocate all quantities
    """
    X, DT_EGM, DT_BF = {}, {}, {}
    X['True'] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
    N=N_true,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
    show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
    print("First compute values on fine grid")
    DT_BF['True'] = true_val['DT']
    #DT_EGM['True'] = X['True'].solve_MPFI('EGM',0,DT_BF['True'][0])
    DT_EGM_diff_true, DT_BF_diff_true, DT_EGM_BF = {}, {}, {}
    data_DT_EGM, data_DT_BF, data_compare = [], [], []
    for N in N_set:
        print("Number of gridpoints:", N)
        d_DT_EGM, d_DT_BF, d_compare = {}, {}, {}
        X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        N=N,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        DT_BF[N] = X[N].solve_PFI('BF')
        DT_EGM[N] = X[N].solve_MPFI('EGM',0,DT_BF[N][0])
        def compare(f1,f2):
            f2_big = np.zeros((X['True'].N[0]-1,X['True'].N[1]-1))
            for j in range(X['True'].N[1]-1):
                f2_big[:,j] = interp1d(X[N].grid[0], f2[:,j],fill_value="extrapolate")(X['True'].grid[0])
            return f1-f2_big
        DT_BF_diff_true[N] = compare(DT_BF['True'][0], DT_BF[N][0]), compare(DT_BF['True'][1], DT_BF[N][1])
        DT_EGM_diff_true[N] = compare(DT_BF['True'][0], DT_EGM[N][0]), compare(DT_BF['True'][1], DT_EGM[N][1])
        DT_EGM_BF[N] = DT_EGM[N][0] - DT_BF[N][0], DT_EGM[N][1] - DT_BF[N][1]
        #compare EGM output with EGM "truth"
        d_DT_EGM[cols[0]] = np.mean(np.abs(DT_EGM_diff_true[N][1]))
        d_DT_EGM[cols[1]] = np.max(np.abs(DT_EGM_diff_true[N][1]))
        d_DT_EGM[cols[2]] = np.mean(np.abs(DT_EGM_diff_true[N][0]))
        d_DT_EGM[cols[3]] = np.max(np.abs(DT_EGM_diff_true[N][0]))
        #compare BF output with BF "truth"
        d_DT_BF[cols[0]] = np.mean(np.abs(DT_BF_diff_true[N][1]))
        d_DT_BF[cols[1]] = np.max(np.abs(DT_BF_diff_true[N][1]))
        d_DT_BF[cols[2]] = np.mean(np.abs(DT_BF_diff_true[N][0]))
        d_DT_BF[cols[3]] = np.max(np.abs(DT_BF_diff_true[N][0]))
        #compare EGM and BF output
        d_compare[cols_compare[0]] = np.mean(np.abs(DT_EGM_BF[N][1]))
        d_compare[cols_compare[1]] = np.max(np.abs(DT_EGM_BF[N][1]))
        d_compare[cols_compare[2]] = np.mean(np.abs(DT_EGM_BF[N][0]))
        d_compare[cols_compare[3]] = np.max(np.abs(DT_EGM_BF[N][0]))
        #append the above results
        data_DT_EGM.append(d_DT_EGM)
        data_DT_BF.append(d_DT_BF)
        data_compare.append(d_compare)
    return data_DT_EGM, data_DT_BF, data_compare

def accuracy_tables(true_val,N_set,DT_dt):
    data_DT_EGM, data_DT_BF, data_compare = accuracy_data(true_val,N_set,DT_dt)

    df = pd.DataFrame(data=data_DT_EGM,index=N_set,columns=cols)
    df = df[cols].round(decimals=n_round_acc)
    df.index.names = ['Grid size']

    destin = '../../main/figures/DT_EGM_accuracy_stat_{0}.tex'.format(int(10**3*DT_dt))
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

    df = pd.DataFrame(data=data_DT_BF,index=N_set,columns=cols)
    df = df[cols].round(decimals=n_round_acc)
    df.index.names = ['Grid size']

    destin = '../../main/figures/DT_BF_accuracy_stat_{0}.tex'.format(int(10**3*DT_dt))
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

    df = pd.DataFrame(data=data_compare,index=N_set,columns=cols_compare)
    df = df[cols_compare].round(decimals=n_round_acc)
    df.index.names = ['Grid size']

    destin = '../../main/figures/DT_EGM_BF_accuracy_stat_{0}.tex'.format(int(10**3*DT_dt))
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

"""
Create tables
"""
DT_dt = 1
true_val = true_DT_stat(DT_dt)
accuracy_tables(true_val, parameters.N_set, DT_dt)
