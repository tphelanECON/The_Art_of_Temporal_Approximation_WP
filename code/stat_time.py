"""
Run times and number of iterations for stationary income fluctuation problems
(both discrete- and continuous-time).

Relevant class constructors in classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent)
    CT_stat_IFP: continuous-time stationary IFP


"""

import numpy as np
import pandas as pd
import time, classes, parameters, json

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mu, sigma = parameters.mu, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS, max_age = parameters.bnd, parameters.bnd_NS, parameters.max_age
show_method, show_iter, show_final = 1, 0, 1
n_round = parameters.n_round

maxiter = 2*10**4
relax_list = [10,50,100]

cols = ['VFI']
for relax in relax_list:
    cols.append(r'MPFI ($k=$ {0})'.format(relax))
cols.append('PFI')

def time_data(N_set,DT_dt,CT_dt,runs):
    df_DT = pd.DataFrame(data=0,index=N_set,columns=cols)
    df_DT_iter = pd.DataFrame(data=0,index=N_set,columns=cols)
    df_CT = pd.DataFrame(data=0,index=N_set,columns=cols)
    df_CT_iter = pd.DataFrame(data=0,index=N_set,columns=cols)
    for i in range(runs):
        time_data_DT, time_data_CT = [], []
        iter_data_DT, iter_data_CT = [], []
        X, Y = {}, {}
        for N in N_set:
            print("Number of gridpoints:", N)
            d_DT, d_CT = {}, {}
            d_DT_iter, d_CT_iter = {}, {}
            X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
            N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
            show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
            Y[N] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
            N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
            show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
            d_DT['PFI'], d_DT_iter['PFI'] = X[N].solve_PFI('EGM')[2:]
            d_CT['PFI'], d_CT_iter['PFI'] = Y[N].solve_PFI()[2:]
            d_DT['VFI'], d_DT_iter['VFI'] = X[N].solve_MPFI('EGM',0,X[N].V0)[2:]
            d_CT['VFI'], d_CT_iter['VFI'] = Y[N].solve_MPFI(0)[2:]
            for relax in relax_list:
                s = r'MPFI ($k = $ {0})'.format(relax)
                d_DT[s], d_DT_iter[s] = X[N].solve_MPFI('EGM',relax,X[N].V0)[2:]
                d_CT[s], d_CT_iter[s] = Y[N].solve_MPFI(relax)[2:]
            time_data_DT.append(d_DT)
            iter_data_DT.append(d_DT_iter)
            time_data_CT.append(d_CT)
            iter_data_CT.append(d_CT_iter)
        df_DT = df_DT + pd.DataFrame(data=time_data_DT,index=N_set,columns=cols)
        df_DT_iter = df_DT_iter + pd.DataFrame(data=iter_data_DT,index=N_set,columns=cols)
        df_CT = df_CT + pd.DataFrame(data=time_data_CT,index=N_set,columns=cols)
        df_CT_iter = df_CT_iter + pd.DataFrame(data=iter_data_CT,index=N_set,columns=cols)
    return df_DT.round(decimals=n_round)/runs, df_CT.round(decimals=n_round)/runs, df_DT_iter, df_CT_iter

def time_tables(N_set,DT_dt,CT_dt,runs):
    df_DT, df_CT, df_DT_iter, df_CT_iter = time_data(N_set,DT_dt,CT_dt,runs)
    df_DT.index.names = ['Grid size']
    df_CT.index.names = ['Grid size']
    df_DT_iter.index.names = ['Grid size']
    df_CT_iter.index.names = ['Grid size']

    destin = '../main/figures/DT_speed_stat_{0}.tex'.format(int(1000*DT_dt))
    with open(destin,'w') as tf:
        tf.write(df_DT.to_latex(escape=False,column_format='cccccc'))

    destin = '../main/figures/CT_speed_stat_{0}.tex'.format(int(1000*CT_dt))
    with open(destin,'w') as tf:
        tf.write(df_CT.to_latex(escape=False,column_format='cccccc'))

    destin = '../main/figures/DT_iter_stat_{0}.tex'.format(int(1000*DT_dt))
    with open(destin,'w') as tf:
        tf.write(df_DT_iter.to_latex(escape=False,column_format='cccccc'))

    destin = '../main/figures/CT_iter_stat_{0}.tex'.format(int(1000*CT_dt))
    with open(destin,'w') as tf:
        tf.write(df_CT_iter.to_latex(escape=False,column_format='cccccc'))

"""
Create tables. Recall arguments: (N_set,DT_dt,CT_dt).
"""

N_set = parameters.N_set
DT_dt = 1.0
CT_dt = 10**-2
runs = 1

time_tables(parameters.N_set,DT_dt,CT_dt,runs)
