"""
Accuracy and run times for non-stationary IFPs. Parallels stat_time_accuracy.py.

Relevant class constructors imported from classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent).
    CT_nonstat_IFP: continuous-time stationary IFP.

Methods:
    * accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF'):
    takes dict true_val of true values, list of grid sizes N_set for assets and
    income, DT time step DT_dt and CT time step CT_dt and returns
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

Policy functions vanish for highest age, so we restrict attention to [:,:,:-1].
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

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 1, 1, 1
NA = parameters.NA
NA_true = parameters.NA_true
N_t = parameters.N_t
N_c = parameters.N_c
n_round_acc = parameters.n_round_acc
n_round_time = parameters.n_round_time
CT_dt_true = parameters.CT_dt_true
DT_dt = parameters.DT_dt

"""
The following are simply fixed throughout.
"""

NA = parameters.NA
NA_scale = parameters.NA_scale
relax_list = parameters.relax_list

cols_true = parameters.cols_true
cols_compare = parameters.cols_compare
cols_time_nonstat = parameters.cols_time_nonstat
cols_time_nonstat_decomp = parameters.cols_time_nonstat_decomp

def accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob='KD'):
    N_true_shape = true_val['DT'][0].shape
    if N_true_shape[1]-1!=N_set[0][1]:
        print("Error: true value has different number of income points than test values")
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
        d_DT, d_CT = {}, {}
        X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N,NA=NA,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        bnd=bnd_NS,N=(N[0],N[1],parameters.NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)
        DT_shape, CT_shape = true_val['DT'][0].shape, true_val['CT'][0].shape
        fine_asset_grid = np.linspace(bnd[0][0],bnd[0][1],N_true_shape[0])
        #f1 true values on fine grid. f2 on coarse grid; interpolated on fine grid.
        #relative to stat, one more loop in following over ages:
        def compare(f1,f2):
            f2_fine = np.zeros((N_true_shape[0],N_true_shape[1],DT_shape[2]))
            for j in range(N_true_shape[1]):
                for k in range(DT_shape[2]):
                    f2_fine[:,j,k] = interp1d(X[N].grid[0], f2[:,j,k],fill_value="extrapolate")(fine_asset_grid)
            return f1-f2_fine
        #compute DT output and compare with truth.
        if framework in ['DT','both']:
            DT[N] = X[N].nonstat_solve(method,prob)[0:2]
            d = np.array(compare(DT['True'][1], DT[N][1]))[:,:,:-1]
            d_percent = 100*d/(DT['True'][1][:,:,:-1])
            d_DT[cols_true[0]] = np.mean(np.abs(d))
            d_DT[cols_true[1]] = np.max(np.abs(d))
            d_DT[cols_true[2]] = np.mean(np.abs(d_percent))
            d_DT[cols_true[3]] = np.max(np.abs(d_percent))
            data_DT.append(d_DT)
        #compute CT output and compare with truth
        if framework in ['CT','both']:
            CT[N] = Z[N].solve_seq_imp()[0:2]
            d = np.array(compare(CT['True'][1], CT[N][1]))[:,:,:-1]
            d_percent = 100*d/(CT['True'][1][:,:,:-1])
            d_CT[cols_true[0]] = np.mean(np.abs(d))
            d_CT[cols_true[1]] = np.max(np.abs(d))
            d_CT[cols_true[2]] = np.mean(np.abs(d_percent))
            d_CT[cols_true[3]] = np.max(np.abs(d_percent))
            data_CT.append(d_CT)
    if framework=='DT':
        return pd.DataFrame(data=data_DT,index=N_set,columns=cols_true)
    elif framework=='CT':
        return pd.DataFrame(data=data_CT,index=N_set,columns=cols_true)
    else:
        return pd.DataFrame(data=data_DT,index=N_set,columns=cols_true), \
        pd.DataFrame(data=data_CT,index=N_set,columns=cols_true)

def comparison_data(N_set,DT_dt,CT_dt,method='EGM',prob='KD'):
    X, Z, DT, CT = {}, {}, {}, {}
    data_compare = []
    for N in N_set:
        print("Number of gridpoints:", N)
        d_compare = {}
        X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N,NA=NA,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        bnd=bnd_NS,N=(N[0],N[1],parameters.NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)
        DT[N] = X[N].nonstat_solve(method,prob)[0:2]
        CT[N] = Z[N].solve_seq_imp()[0:2]
        d = (DT[N][1]-CT[N][1])[:,:,:-1]
        d_percent = 100*d/(CT[N][1][:,:,:-1])
        d_compare[cols_compare[0]] = np.mean(np.abs(d))
        d_compare[cols_compare[1]] = np.max(np.abs(d))
        d_compare[cols_compare[2]] = np.mean(np.abs(d_percent))
        d_compare[cols_compare[3]] = np.max(np.abs(d_percent))
        data_compare.append(d_compare)
    return pd.DataFrame(data=data_compare,index=N_set,columns=cols_compare)

def accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob='KD'):
    N_I = N_set[0][1]
    if framework=='both':
        df_DT, df_CT = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method,prob)
    if framework=='DT':
        df_DT = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method,prob)
    if framework=='CT':
        df_CT = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method,prob)

    if framework in ['DT','both']:
        df = pd.DataFrame(data=df_DT,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/DT_{0}_accuracy_nonstat_{1}_{2}_{3}.tex'.format(method,int(10**3*DT_dt),prob,N_I)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    if framework in ['CT','both']:
        df = pd.DataFrame(data=df_CT,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/CT_accuracy_nonstat_{0}_{1}_{2}.tex'.format(int(10**6*CT_dt),prob,N_I)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

def comparison_tables(N_set,DT_dt,CT_dt,method='EGM',prob='KD'):
    N_I = N_set[0][1]
    df_compare = comparison_data(N_set,DT_dt,CT_dt,method,prob)
    df = pd.DataFrame(data=df_compare,index=N_set,columns=cols_compare)
    df = df[cols_compare].round(decimals=n_round_acc)
    df.index.names = ['Grid size']

    destin = '../../main/figures/DT_CT_{0}_accuracy_nonstat_{1}_{2}_{3}_{4}.tex'.format(method,int(10**3*DT_dt),int(10**6*CT_dt),prob,N_I)
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_compare)+1)))

#no need for framework in the following. only use three candidates.
def time_data(N_set,DT_dt,CT_dt,runs,prob='KD',run_PFI=True):
    df = pd.DataFrame(data=0,index=N_set,columns=cols_time_nonstat)
    df_decomp = pd.DataFrame(data=0,index=N_set,columns=cols_time_nonstat_decomp)
    for i in range(runs):
        #cols_time_nonstat_decomp = ['DT policy','DT value','CT policy','CT value']
        time_data, time_data_decomp = [], []
        X, Z = {}, {}
        for N in N_set:
            print("Number of gridpoints:", N)
            d, d_decomp = {}, {}
            X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
            N=N,NA=NA,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
            show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
            Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
            bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
            tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)
            sol_DT = X[N].nonstat_solve('EGM',prob)
            sol_CT = Z[N].solve_seq_imp()
            d[cols_time_nonstat[0]] = sol_DT[2]
            d_decomp[cols_time_nonstat_decomp[0]] = sol_DT[3].sum(axis=1)[0]
            d_decomp[cols_time_nonstat_decomp[1]] = sol_DT[3].sum(axis=1)[1]
            d[cols_time_nonstat[1]] = sol_CT[2]
            d_decomp[cols_time_nonstat_decomp[2]] = sol_CT[3].sum(axis=1)[0]
            d_decomp[cols_time_nonstat_decomp[3]] = sol_CT[3].sum(axis=1)[1]
            if (N[0] < 200) and (run_PFI==True):
                d[cols_time_nonstat[2]] = Z[N].solve_PFI()[2]
            else:
                d[cols_time_nonstat[2]] = np.Inf
            time_data.append(d)
            time_data_decomp.append(d_decomp)
        df = df + pd.DataFrame(data=time_data,index=N_set,columns=cols_time_nonstat)
        df_decomp = df_decomp + pd.DataFrame(data=time_data_decomp,index=N_set,columns=cols_time_nonstat_decomp)
    return df.round(decimals=n_round_time)/runs, df_decomp.round(decimals=n_round_time)/runs

def time_tables(N_set,DT_dt,CT_dt,runs,prob='KD'):
    N_I = N_set[0][1]
    data = time_data(N_set,DT_dt,CT_dt,runs,prob,run_PFI=True)
    df, df_decomp = data
    df.index.names, df_decomp.index.names = ['Grid size'], ['Grid size']

    destin = '../../main/figures/DT_CT_speed_nonstat_{0}_{1}_{2}_{3}.tex'.format(int(1000*DT_dt),int(10**6*CT_dt),prob,N_I)
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time_nonstat)+1)))

    destin = '../../main/figures/DT_CT_speed_nonstat_{0}_{1}_{2}_{3}_decomp.tex'.format(int(1000*DT_dt),int(10**6*CT_dt),prob,N_I)
    with open(destin,'w') as tf:
        tf.write(df_decomp.to_latex(escape=False,column_format='c'*(len(cols_time_nonstat)+1)))

"""
Use EGM for accuracy in following (that is what is used in time tables).

No max over relaxations in following because MPFI not used in nonstat setting.

Also run_PFI=False because naive PFI is never a good idea.
"""
def time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob='KD',norm='mean'):
    DT, CT = {}, {}
    DT['accuracy'], CT['accuracy'] = [], []
    DT['time'], CT['time'] = [], []
    df_acc_DT, df_acc_CT = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob=prob)
    df_time = time_data(N_set,DT_dt,CT_dt,runs,prob,run_PFI=False)[0]

    #remember that we always care about percent deviations of policy functions
    for i in range(len(N_set)):
        if norm=='mean':
            DT['accuracy'].append(df_acc_DT.iloc[i,2])
            CT['accuracy'].append(df_acc_CT.iloc[i,2])
        else:
            DT['accuracy'].append(df_acc_DT.iloc[i,3])
            CT['accuracy'].append(df_acc_CT.iloc[i,3])
        DT['time'].append(df_time.iloc[i,0])
        CT['time'].append(df_time.iloc[i,1])
    return DT, CT

def time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs,prob='KD',norm='mean'):
    N_I = N_set[0][1]
    DT, CT = time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob,norm)
    fig, ax = plt.subplots()
    ax.scatter(DT['accuracy'],DT['time'],marker='x',color='darkblue',lw=2,label='Discrete-time')
    ax.scatter(CT['accuracy'],CT['time'],marker='x',color='lightsteelblue',lw=2,label='Continuous-time')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylabel("Time to convergence (seconds)")
    plt.xlabel("Percent deviation from true policy function")
    plt.legend()
    if norm=='mean':
        ax.set_title('$l_1$ norm' + ' ({0} income points)'.format(N_I))
    else:
        ax.set_title('$l_{\infty}$ norm' + ' ({0} income points)'.format(N_I))
    destin = '../../main/figures/time_accuracy_nonstat_{0}_{1}_{2}_{3}_{4}.eps'.format(int(10**3*DT_dt),int(10**6*CT_dt),prob,norm,N_I)
    plt.savefig(destin, format='eps', dpi=1000)
    #plt.show()
    plt.close()

"""
Create tables. Unlike stationary case, no need to loop over different income.
"""
i=1
N = parameters.N_true_set[i]
N_set = parameters.N_sets[i]
N_scatter = parameters.N_sets_scatter[i]
"""
Accuracy
"""
true_val = true_nonstat_load(parameters.DT_dt, parameters.CT_dt_true, parameters.NA, 'KD', N)
accuracy_tables(true_val, parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_true, 'both', 'EGM', 'KD')
"""
Time for convergence
"""
runs=10
for prob in ['KD']:
    time_tables(parameters.N_sets[0], parameters.DT_dt, parameters.CT_dt_true, runs, prob)
    time_tables(parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_true, runs, prob)
"""
Speed versus accuracy
"""
for prob in ['KD']:
    true_val = true_nonstat_load(parameters.DT_dt, parameters.CT_dt_true, parameters.NA, prob, N)
    time_accuracy_figures(true_val, N_scatter, parameters.DT_dt, parameters.CT_dt_true, runs, prob,'mean')
    time_accuracy_figures(true_val, N_scatter, parameters.DT_dt, parameters.CT_dt_true, runs, prob,'sup')
