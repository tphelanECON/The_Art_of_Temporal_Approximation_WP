"""
Comparison for brute force method with the continuous-time method, both between
each other and with respect to the "true" values.

Relevant class constructors in classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent)
    CT_stat_IFP: continuous-time stationary IFP

Al solve functions (solve_MPFI, solve_PFI) in above class constructors
return quadruples (V, c, toc-tic, i) where i = no. of iterations.

functions:
    1. accuracy_data(N_set,DT_dt,CT_dt); and
    2. accuracy_tables(N_set,DT_dt,CT_dt).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import time, classes, parameters
import matplotlib.pyplot as plt

c1,c2 = parameters.c1, parameters.c2
colorFader = parameters.colorFader

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mu, sigma = parameters.mu, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 1, 1, 1
N_true, N_c, n_round = parameters.N_true, parameters.N_c, 4

cols = ['$||c - c_{\textnormal{true}}||_1$',
'$||c - c_{\textnormal{true}}||_{\infty}$',
'$||V - V_{\textnormal{true}}||_1$',
'$||V - V_{\textnormal{true}}||_{\infty}$']

cols_compare = ['$||c_{DT} - c_{CT}||_1$',
'$||c_{DT} - c_{CT}||_{\infty}$',
'$||V_{DT} - V_{CT}||_1$',
'$||V_{DT} - V_{CT}||_{\infty}$']

"""
Functions for data construction and table creation
"""

def accuracy_data(N_set,DT_dt,CT_dt):
    """
    Pre-allocate all quantities
    """
    X, Y, DT, CT = {}, {}, {}, {}
    X['True'] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
    N=N_true,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
    show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
    Y['True'] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
    N=N_true,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
    show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
    print("First compute values on fine grid")
    CT['True'] = Y['True'].solve_PFI()
    DT['True'] = X['True'].solve_PFI('BF')
    DT_diff_true, CT_diff_true, DT_CT = {}, {}, {}
    data_DT, data_CT, data_compare = [], [], []
    for N in N_set:
        print("Number of gridpoints:", N)
        d_DT, d_CT, d_compare = {}, {}, {}
        X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        N=N,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        Y[N] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        CT[N] = Y[N].solve_PFI()
        DT[N] = X[N].solve_PFI('BF')
        def compare(f1,f2):
            f2_big = np.zeros((X['True'].N[0]-1,X['True'].N[1]-1))
            for j in range(X['True'].N[1]-1):
                f2_big[:,j] = interp1d(X[N].grid[0], f2[:,j],fill_value="extrapolate")(X['True'].grid[0])
            return f1-f2_big
        DT_diff_true[N] = compare(DT['True'][0], DT[N][0]), compare(DT['True'][1], DT[N][1])
        CT_diff_true[N] = compare(CT['True'][0], CT[N][0]), compare(CT['True'][1], CT[N][1])
        DT_CT[N] = DT[N][0]-CT[N][0], DT[N][1]-CT[N][1]
        #compare discrete-time output with discrete-time "truth"
        d_DT[cols[0]] = np.mean(np.abs(DT_diff_true[N][1]))
        d_DT[cols[1]] = np.max(np.abs(DT_diff_true[N][1]))
        d_DT[cols[2]] = np.mean(np.abs(DT_diff_true[N][0]))
        d_DT[cols[3]] = np.max(np.abs(DT_diff_true[N][0]))
        #compare continuous-time output with continuous-time "truth"
        d_CT[cols[0]] = np.mean(np.abs(CT_diff_true[N][1]))
        d_CT[cols[1]] = np.max(np.abs(CT_diff_true[N][1]))
        d_CT[cols[2]] = np.mean(np.abs(CT_diff_true[N][0]))
        d_CT[cols[3]] = np.max(np.abs(CT_diff_true[N][0]))
        #compare discrete- and continuous-time output
        d_compare[cols_compare[0]] = np.mean(np.abs(DT_CT[N][1]))
        d_compare[cols_compare[1]] = np.max(np.abs(DT_CT[N][1]))
        d_compare[cols_compare[2]] = np.mean(np.abs(DT_CT[N][0]))
        d_compare[cols_compare[3]] = np.max(np.abs(DT_CT[N][0]))
        #append the above results
        data_DT.append(d_DT)
        data_CT.append(d_CT)
        data_compare.append(d_compare)
    return data_DT, data_CT, data_compare

def accuracy_tables(N_set,DT_dt,CT_dt):
    data_DT, data_CT, data_compare = accuracy_data(N_set,DT_dt,CT_dt)

    df = pd.DataFrame(data=data_DT,index=N_set,columns=cols)
    df = df[cols].round(decimals=n_round)
    df.index.names = ['Grid size']

    destin = '../main/figures/DT_accuracy_stat_{0}.tex'.format(int(10**3*DT_dt))
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

    df = pd.DataFrame(data=data_CT,index=N_set,columns=cols)
    df = df[cols].round(decimals=n_round)
    df.index.names = ['Grid size']

    destin = '../main/figures/CT_accuracy_stat_{0}.tex'.format(int(10**6*CT_dt))
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

    df = pd.DataFrame(data=data_compare,index=N_set,columns=cols_compare)
    df = df[cols_compare].round(decimals=n_round)
    df.index.names = ['Grid size']

    destin = '../main/figures/DT_CT_accuracy_stat_{0}_{1}.tex'.format(int(10**3*DT_dt), int(10**6*CT_dt))
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

"""
Create tables. Recall arguments: (N_set,DT_dt,CT_dt).
"""

CT_dt = 10**-6
accuracy_tables(parameters.N_set, 1, CT_dt)
accuracy_tables(parameters.N_set, 10**-1, CT_dt)
accuracy_tables(parameters.N_set, 10**-2, CT_dt)
