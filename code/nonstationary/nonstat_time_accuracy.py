"""
This script computes accuracy and run times for the non-stationary income fluctuation problem.
Scructure largely parallels the analysis in stat_time_accuracy.py.

Relevant class constructors imported from classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent).
    CT_nonstat_IFP: continuous-time stationary IFP.

functions:

true_nonstat has an additional argument

    * true_nonstat(DT_dt, CT_dt, NA): gets the "true" discrete-time and continuous-time
    value functions and policy functions (these are computed and saved elsewhere).
    Returns dictionary with keys ['DT', 'CT'] and for each returns V, c tuple.
    * accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF'):
    takes dictionary true_val of true values, list of grid sizes N_set for
    assets and income, DT time step DT_dt and CT time step CT_dt and returns
    either one or three dataframes, indicating distance between computed and "true"
    values, or difference between the computed quantities across DT and CT.
    For DT also specify policy updating method, 'BF' or 'EGM'.
    * accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF'):
    produces tables using accuracy_data.
    * time_data(N_set,DT_dt,CT_dt,runs,framework='DT',suppress_VFI=False):
    computes time necessary to solve problems on grids given by N_set, for DT and CT.
    Averages over 'runs' number of runs. suppress_VFI avoids performing VFI.
    * time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs): creates tables using time_data.
    * time_tables(true_val,N_set,DT_dt,CT_dt,runs,framework='DT'): creates data
    necessary for scatterplt.
    * time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs): creates the scatterplot.
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
from scipy.interpolate import RegularGridInterpolator
from true_nonstat import true_nonstat_load

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

"""
The following are simply fixed throughout.
"""

NA = parameters.NA
NA_scale = parameters.NA_scale
NA_true = parameters.NA_true


DT_dt = 10**0
CT_dt = CT_dt_true
local_NA = NA
true_val = true_nonstat_load(DT_dt, CT_dt_true, local_NA)


X_true = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
N=N_true,NA=NA,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)

Z_true = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
bnd=bnd_NS,N=(N_true[0],N_true[1],local_NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)

cols_true = ['$||c - c_{\textnormal{true}}||_1$',
'$||c - c_{\textnormal{true}}||_{\infty}$',
'$||V - V_{\textnormal{true}}||_1$',
'$||V - V_{\textnormal{true}}||_{\infty}$']

cols_compare = ['$||c_{DT} - c_{CT}||_1$',
'$||c_{DT} - c_{CT}||_{\infty}$',
'$||V_{DT} - V_{CT}||_1$',
'$||V_{DT} - V_{CT}||_{\infty}$']

cols_time = ['EGM','Seq. PFI','Naive PFI']

def accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM'):
    X, Z, DT, CT = {}, {}, {}, {}
    if framework in ['DT','both']:
        DT['True'] = true_val['DT']
        DT_diff_true, data_DT = {}, []
    if framework in ['CT','both']:
        CT['True'] = true_val['CT']
        CT_diff_true, data_CT = {}, []
    if framework=='both':
        DT_CT, data_compare = {}, []
    for N in N_set:
        print("Number of gridpoints:", N)
        d_DT, d_CT, d_compare = {}, {}, {}
        X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        N=N,NA=NA,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)
        NA_scale_loc = int(local_NA/NA)
        def compare(f1,f2,framework):
            if framework == 'DT':
                f2_fine = np.zeros((X_true.N[0]-1, X_true.N[1]-1, X_true.NA-1))
                for k in range(X_true.NA-1):
                    for j in range(X_true.N[1]-1):
                        f2_fine[:,j,k] = interp1d(X[N].grid[0], f2[:,j,k], fill_value="extrapolate")(X_true.grid[0])
                return f1-f2_fine
            if framework == 'CT':
                #f1 is the true CT case. Want to just evaluate this on the discrete-time age grid.
                f1_coarse = np.zeros((Z_true.N[0]-1,Z_true.N[1]-1,NA-1))
                f2_fine = np.zeros((Z_true.N[0]-1,Z_true.N[1]-1,NA-1))
                for k in range(NA-1):
                    f1_coarse[:,:,k] = f1[:,:,(k+1)*NA_scale_loc-1]
                    for j in range(Z_true.N[1]-1):
                        f2_fine[:,j,k] = interp1d(Z[N].grid[0], f2[:,j,k], fill_value="extrapolate")(Z_true.grid[0])
                return f1_coarse-f2_fine
        #compute DT output and compare with truth.
        if framework in ['DT','both']:
            DT[N] = X[N].nonstat_solve(method)[0:2]
            d = np.array(compare(DT['True'][0],DT[N][0],'DT')), np.array(compare(DT['True'][1],DT[N][1],'DT'))
            d_DT[cols_true[0]] = np.mean(np.abs(d[1]))
            d_DT[cols_true[1]] = np.max(np.abs(d[1]))
            d_DT[cols_true[2]] = np.mean(np.abs(d[0]))
            d_DT[cols_true[3]] = np.max(np.abs(d[0]))
            data_DT.append(d_DT)
        #compute CT output and compare with truth
        if framework in ['CT','both']:
            CT[N] = Z[N].solve_seq_imp()[0:2]
            d = np.array(compare(CT['True'][0],CT[N][0],'CT')), np.array(compare(CT['True'][1], CT[N][1],'CT'))
            d_CT[cols_true[0]] = np.mean(np.abs(d[1]))
            d_CT[cols_true[1]] = np.max(np.abs(d[1]))
            d_CT[cols_true[2]] = np.mean(np.abs(d[0]))
            d_CT[cols_true[3]] = np.max(np.abs(d[0]))
            data_CT.append(d_CT)
        #compare across DT and CT
        if framework=='both':
            d = DT[N][0]-CT[N][0], DT[N][1]-CT[N][1]
            d_compare[cols_compare[0]] = np.mean(np.abs(d[1]))
            d_compare[cols_compare[1]] = np.max(np.abs(d[1]))
            d_compare[cols_compare[2]] = np.mean(np.abs(d[0]))
            d_compare[cols_compare[3]] = np.max(np.abs(d[0]))
            data_compare.append(d_compare)
    if framework=='DT':
        return pd.DataFrame(data=data_DT,index=N_set,columns=cols_true)
    elif framework=='CT':
        return pd.DataFrame(data=data_CT,index=N_set,columns=cols_true)
    else:
        return pd.DataFrame(data=data_DT,index=N_set,columns=cols_true), \
        pd.DataFrame(data=data_CT,index=N_set,columns=cols_true), \
        pd.DataFrame(data=data_compare,index=N_set,columns=cols_compare)

def accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM'):
    if framework=='DT':
        df_data = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method)
        df = pd.DataFrame(data=df_data,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/DT_accuracy_nonstat_{0}.tex'.format(int(10**3*DT_dt))
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    elif framework=='CT':
        df_data = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method)
        df = pd.DataFrame(data=df_data,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/CT_accuracy_nonstat_{0}.tex'.format(int(10**6*CT_dt))
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    elif framework=='both':
        df_DT, df_CT, df_compare = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method)
        df = pd.DataFrame(data=df_compare,index=N_set,columns=cols_compare)
        df = df[cols_compare].round(decimals=n_round_acc)
        df.index.names = ['Grid size']

        destin = '../../main/figures/DT_CT_accuracy_nonstat_{0}_{1}.tex'.format(int(10**3*DT_dt), int(10**6*CT_dt))
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_compare)+1)))

def time_data(N_set,DT_dt,CT_dt,runs, suppress_PFI=False):
    df = pd.DataFrame(data=0,index=N_set,columns=cols_time)
    for i in range(runs):
        time_data = []
        X, Z = {}, {}
        for N in N_set:
            print("Number of gridpoints:", N)
            d = {}
            X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
            N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
            show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
            Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
            bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
            tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)
            d[cols_time[0]] = X[N].nonstat_solve('EGM')[2]
            d[cols_time[1]] = Z[N].solve_seq_imp()[2]
            if (N[0] < 10) and (suppress_PFI==False):
                d[cols_time[2]] = Z[N].solve_PFI()[2]
            else:
                d[cols_time[2]] = np.Inf
            time_data.append(d)
        df = df + pd.DataFrame(data=time_data,index=N_set,columns=cols_time)
    return df.round(decimals=n_round_time)/runs

def time_tables(true_val,N_set,DT_dt,CT_dt,runs):
    df = time_data(N_set,DT_dt,CT_dt,runs)
    df.index.names = ['Grid size']

    destin = '../../main/figures/DT_CT_speed_nonstat_{0}_{1}.tex'.format(int(1000*DT_dt),int(10**6*CT_dt))
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

"""
Use EGM for the accuracy in the following, because that is what is used in the time tables.
"""
def time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs):
    DT, CT = {}, {}
    DT['accuracy'], CT['accuracy'] = [], []
    DT['time'], CT['time'] = [], []
    df_acc_DT, df_acc_CT, df_acc_compare = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM')
    df_time = time_data(N_set,DT_dt,CT_dt,runs,suppress_PFI=True)
    for i in range(len(N_set)):
        DT['accuracy'].append(df_acc_DT.iloc[i,0])
        CT['accuracy'].append(df_acc_CT.iloc[i,0])
        DT['time'].append(df_time.iloc[i,0])
        CT['time'].append(df_time.iloc[i,1])
    return DT, CT

def time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs):
    DT, CT = time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs)
    fig, ax = plt.subplots()
    ax.scatter(DT['accuracy'],DT['time'],marker='x',color='darkblue',lw=2,label='Discrete-time')
    ax.scatter(CT['accuracy'],CT['time'],marker='x',color='lightsteelblue',lw=2,label='Continuous-time')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylabel("Time to convergence (seconds)")
    plt.xlabel("Accuracy")
    plt.legend()
    ax.set_title('Speed versus accuracy tradeoff')
    destin = '../../main/figures/time_accuracy_nonstat_{0}_{1}.eps'.format(int(10**3*DT_dt),int(10**6*CT_dt))
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

"""
Create tables
"""

runs = 1
N_grid = np.linspace(np.log(50), np.log(1000), 10)
N_set_big = [(int(np.rint(np.exp(n))),10) for n in N_grid]

accuracy_tables(true_val, parameters.N_set, DT_dt, CT_dt_true, 'both', method='EGM')
accuracy_tables(true_val, parameters.N_set, DT_dt, CT_dt_true, 'DT', method='EGM')
accuracy_tables(true_val, parameters.N_set, DT_dt, CT_dt_true, 'CT', method='EGM')
time_tables(true_val, parameters.N_set, DT_dt, CT_dt_true, runs)
time_accuracy_figures(true_val, N_set_big, DT_dt, CT_dt_true, runs)
