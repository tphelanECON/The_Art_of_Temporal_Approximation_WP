"""
This script computes accuracy and run times for the non-stationary income fluctuation problem.

Scructure largely parallels the analysis in stat_time_accuracy.py. Make sure
that this last expression is stable first.

Relevant class constructors imported from classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent).
    CT_nonstat_IFP: continuous-time stationary IFP.

Methods:

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
mubar, sigma = parameters.mubar, parameters.sigma
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
relax_list = parameters.relax_list

cols_true = parameters.cols_true
cols_compare = parameters.cols_compare
cols_time_nonstat = parameters.cols_time_nonstat

def accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob='KD'):
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
        X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N,NA=parameters.NA,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        bnd=bnd_NS,N=(N[0],N[1],parameters.NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)
        DT_shape, CT_shape = true_val['DT'][0].shape, true_val['CT'][0].shape
        local_NA_true = CT_shape[2]+1
        Delta = (bnd[0][1]-bnd[0][0])/N_true[0]
        fine_asset_grid = np.linspace(bnd[0][0]+Delta,bnd[0][1]-Delta,N_true[0]-1)
        NA_scale_loc = int(local_NA_true/parameters.NA)
        def compare(f1,f2,framework):
            if framework == 'DT':
                f2_fine = 0*f1 #create fine version of f2 (as fine as f1)
                for k in range(DT_shape[2]):
                    for j in range(DT_shape[1]):
                        f2_fine[:,j,k] = interp1d(X[N].grid[0], f2[:,j,k], fill_value="extrapolate")(fine_asset_grid)
                return f1-f2_fine
            if framework == 'CT':
                #two steps here (potentially). restrict true to coarse age grid,
                #the interpolate coarse on the fine asset grid.
                #f1 is the true CT case. Want to just evaluate this on the discrete-time age grid.
                f1_coarse_age = np.zeros((CT_shape[0],CT_shape[1],parameters.NA-1))
                f2_fine = np.zeros((CT_shape[0],CT_shape[1],parameters.NA-1))
                for k in range(parameters.NA-1):
                    f1_coarse_age[:,:,k] = f1[:,:,(k+1)*NA_scale_loc-1]
                    for j in range(CT_shape[1]):
                        f2_fine[:,j,k] = interp1d(Z[N].grid[0], f2[:,j,k], fill_value="extrapolate")(fine_asset_grid)
                return f1_coarse_age-f2_fine
        #compute DT output and compare with truth.
        if framework in ['DT','both']:
            DT[N] = X[N].nonstat_solve(method,prob)[0:2]
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

def accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob='KD'):
    #get data once
    if framework=='both':
        df_DT, df_CT, df_compare = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method,prob)
    if framework=='DT':
        df_DT = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method,prob)
    if framework=='CT':
        df_CT = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method,prob)

    if framework in ['DT','both']:
        df = pd.DataFrame(data=df_DT,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/DT_{0}_accuracy_nonstat_{1}_{2}.tex'.format(method,int(10**3*DT_dt),prob)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    if framework in ['CT','both']:
        df = pd.DataFrame(data=df_CT,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/CT_accuracy_nonstat_{0}_{1}.tex'.format(int(10**6*CT_dt),prob)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    if framework=='both':
        df = pd.DataFrame(data=df_compare,index=N_set,columns=cols_compare)
        df = df[cols_compare].round(decimals=n_round_acc)
        df.index.names = ['Grid size']

        destin = '../../main/figures/DT_CT_{0}_accuracy_nonstat_{1}_{2}_{3}.tex'.format(method,int(10**3*DT_dt),int(10**6*CT_dt),prob)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_compare)+1)))

#no need for framework in the following. only use three candidates.
def time_data(N_set,DT_dt,CT_dt,runs,prob='KD',run_PFI=True):
    df = pd.DataFrame(data=0,index=N_set,columns=cols_time_nonstat)
    for i in range(runs):
        time_data = []
        X, Z = {}, {}
        for N in N_set:
            print("Number of gridpoints:", N)
            d = {}
            X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
            N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
            show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
            Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
            bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
            tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)
            d[cols_time_nonstat[0]] = X[N].nonstat_solve('EGM',prob)[2]
            d[cols_time_nonstat[1]] = Z[N].solve_seq_imp()[2]
            if (N[0] < 100) and (run_PFI==True):
                d[cols_time_nonstat[2]] = Z[N].solve_PFI()[2]
            else:
                d[cols_time_nonstat[2]] = np.Inf
            time_data.append(d)
        df = df + pd.DataFrame(data=time_data,index=N_set,columns=cols_time_nonstat)
    return df.round(decimals=n_round_time)/runs

def time_tables(true_val,N_set,DT_dt,CT_dt,runs,prob='KD'):
    df = time_data(N_set,DT_dt,CT_dt,runs,prob)
    df.index.names = ['Grid size']

    destin = '../../main/figures/DT_CT_speed_nonstat_{0}_{1}_{2}.tex'.format(int(1000*DT_dt),int(10**6*CT_dt),prob)
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time_nonstat)+1)))

"""
Use EGM for the accuracy in the following, because that is what is used in the time tables.
"""
def time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob='KD'):
    DT, CT = {}, {}
    DT['accuracy'], CT['accuracy'] = [], []
    DT['time'], CT['time'] = [], []
    df_acc_DT, df_acc_CT, df_acc_compare = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob=prob)
    df_time = time_data(N_set,DT_dt,CT_dt,runs,prob,run_PFI=False)
    for i in range(len(N_set)):
        DT['accuracy'].append(df_acc_DT.iloc[i,0])
        CT['accuracy'].append(df_acc_CT.iloc[i,0])
        DT['time'].append(df_time.iloc[i,0])
        CT['time'].append(df_time.iloc[i,1])
    return DT, CT

def time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs,prob='KD'):
    DT, CT = time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob)
    fig, ax = plt.subplots()
    ax.scatter(DT['accuracy'],DT['time'],marker='x',color='darkblue',lw=2,label='Discrete-time')
    ax.scatter(CT['accuracy'],CT['time'],marker='x',color='lightsteelblue',lw=2,label='Continuous-time')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylabel("Time to convergence (seconds)")
    plt.xlabel("Accuracy")
    plt.legend()
    ax.set_title('Speed versus accuracy tradeoff')
    destin = '../../main/figures/time_accuracy_nonstat_{0}_{1}_{2}.eps'.format(int(10**3*DT_dt),int(10**6*CT_dt),prob)
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

"""
Create tables
"""

runs=10
DT_dt = 10**0
CT_dt = CT_dt_true
prob='KD'
true_val = true_nonstat_load(DT_dt, CT_dt_true, parameters.NA, prob)
N_set = parameters.N_set
accuracy_tables(true_val, N_set, DT_dt, CT_dt_true, 'both', 'BF', prob)
"""
EGM accuracy
"""
true_val = true_nonstat_load(DT_dt, CT_dt_true, parameters.NA, prob)
accuracy_tables(true_val, N_set, DT_dt, CT_dt_true, 'DT', 'EGM', prob)
"""
Time (and number of iterations) for convergence (need to reload original true)
"""
true_val = true_nonstat_load(DT_dt, CT_dt_true, parameters.NA, prob)
time_tables(true_val, N_set, DT_dt, CT_dt_true, runs, prob)
"""
Speed versus accuracy (both KD and Tauchen)
"""
N_grid = np.linspace(np.log(50), np.log(1000), 12)
N_set_big = [(int(np.rint(np.exp(n))),10) for n in N_grid]
for prob in ['KD','Tauchen']:
    true_val = true_nonstat_load(DT_dt, CT_dt_true, parameters.NA, prob=prob)
    time_accuracy_figures(true_val, N_set_big, DT_dt, CT_dt_big, runs, prob=prob)
