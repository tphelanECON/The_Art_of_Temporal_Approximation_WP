"""
Accuracy and run times for stationary IFPs.

Relevant class constructors imported from classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent).
    CT_stat_IFP: continuous-time stationary IFP.

All solve functions (solve_MPFI, solve_PFI) in above class constructors
return quadruples (V, c, toc-tic, i) where i = no. of iterations.

Methods:
    * accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method,prob)
    true_val = dict of true values, N_set = list of grids for (assets, income).
    Returns one or two dfs indicating diff between computed and "true" values.
    * comparison_data(N_set,DT_dt,CT_dt,method,prob) compare output across DT, CT.
    * accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework,method,prob) produces
    tables using accuracy_data.
    * comparison_tables(N_set,DT_dt,CT_dt,method,prob): tables for comparison_data.
    * time_data(N_set,DT_dt,CT_dt,runs,framework,prob,run_MPFI=True) computes
    average run times per grids in N_set. run_MPFI=False avoids performing VFI.
    * time_tables(N_set,DT_dt,CT_dt,runs,framework='DT',prob='KD') tables
    for time taken to solve problems on various grids.
    * time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob,norm): data for time_accuracy_figures.
    * time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs,prob) creates scatterplots.

May 2023: might need to add plt.tight_layout() to ensure readability of figures.
See: https://stackoverflow.com/questions/6774086/how-to-adjust-padding-with-cutoff-or-overlapping-labels
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
if not os.path.exists('../../main/true_values'):
    os.makedirs('../../main/true_values')
import true_stat
from true_stat import true_stat_load

c1, c2 = parameters.c1, parameters.c2
colorFader = parameters.colorFader

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 0, 0, 0
N_t = parameters.N_t
N_c = parameters.N_c

n_round_acc = parameters.n_round_acc
n_round_time = parameters.n_round_time
relax_list = parameters.relax_list
cols_true = parameters.cols_true
cols_compare = parameters.cols_compare
cols_time = parameters.cols_time

"""
Compare with true values. Accuracy notion: percentage deviations of policy function.
"""

def accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob='KD'):
    N_true_shape = true_val['DT'][0].shape
    N_I = N_set[0][1]
    if N_true_shape[1]-1!=N_I:
        print("Error: true value has different number of income points than test values")
    X, Y, DT, CT = {}, {}, {}, {}
    if framework in ['DT','both']:
        DT['True'] = true_val['DT']
        data_DT = []
    if framework in ['CT','both']:
        CT['True'] = true_val['CT']
        data_CT = []
    for N in N_set:
        print("Number of gridpoints:", N)
        d_DT, d_CT, d_compare = {}, {}, {}
        X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        Y[N] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        fine_asset_grid = np.linspace(bnd[0][0],bnd[0][1],N_true_shape[0])
        #f1 true values on fine grid. f2 on coarse grid; interpolated on fine grid.
        def compare(f1,f2):
            f2_fine = np.zeros((N_true_shape[0],N_true_shape[1]))
            for j in range(N_true_shape[1]):
                f2_fine[:,j] = interp1d(X[N].grid[0], f2[:,j],fill_value="extrapolate")(fine_asset_grid)
            return f1-f2_fine
        #compute DT output and compare with truth
        if framework in ['DT','both']:
            DT[N] = X[N].solve_PFI(method,prob)
            d = np.array(compare(DT['True'][1], DT[N][1]))
            d_percent = 100*d/DT['True'][1]
            d_DT[cols_true[0]] = np.mean(np.abs(d))
            d_DT[cols_true[1]] = np.max(np.abs(d))
            d_DT[cols_true[2]] = np.mean(np.abs(d_percent))
            d_DT[cols_true[3]] = np.max(np.abs(d_percent))
            data_DT.append(d_DT)
        #compute CT output and compare with truth
        if framework in ['CT','both']:
            CT[N] = Y[N].solve_PFI()
            d = np.array(compare(CT['True'][1], CT[N][1]))
            d_percent = 100*d/CT['True'][1]
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
        return pd.DataFrame(data=data_DT,index=N_set,columns=cols_true), pd.DataFrame(data=data_CT,index=N_set,columns=cols_true)

def comparison_data(N_set,DT_dt,CT_dt,method='EGM',prob='KD'):
    X, Y, DT, CT = {}, {}, {}, {}
    data_compare = []
    for N in N_set:
        print("Number of gridpoints:", N)
        d_compare = {}
        X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        Y[N] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        DT[N] = X[N].solve_PFI(method,prob)
        CT[N] = Y[N].solve_PFI()
        d = DT[N][1]-CT[N][1]
        d_percent = 100*d/CT[N][1]
        d_compare[cols_compare[0]] = np.mean(np.abs(d))
        d_compare[cols_compare[1]] = np.max(np.abs(d))
        d_compare[cols_compare[2]] = np.mean(np.abs(d_percent))
        d_compare[cols_compare[3]] = np.max(np.abs(d_percent))
        data_compare.append(d_compare)
    return pd.DataFrame(data=data_compare,index=N_set,columns=cols_compare)

#destinations in following are indexed by number of income points
def accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF',prob='KD'):
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
        destin = '../../main/figures/DT_{0}_accuracy_stat_{1}_{2}_{3}.tex'.format(method,int(10**3*DT_dt),prob,N_I)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    if framework in ['CT','both']:
        df = pd.DataFrame(data=df_CT,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/CT_accuracy_stat_{0}_{1}_{2}.tex'.format(int(10**6*CT_dt),prob,N_I)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

def comparison_tables(N_set,DT_dt,CT_dt,method='EGM',prob='KD'):
    N_I = N_set[0][1]
    df_compare = comparison_data(N_set,DT_dt,CT_dt,method,prob)
    df = pd.DataFrame(data=df_compare,index=N_set,columns=cols_compare)
    df = df[cols_compare].round(decimals=n_round_acc)
    df.index.names = ['Grid size']

    destin = '../../main/figures/DT_CT_{0}_accuracy_stat_{1}_{2}_{3}_{4}.tex'.format(method,int(10**3*DT_dt),int(10**6*CT_dt),prob,N_I)
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_compare)+1)))

#following only uses EGM; no need for 'method' argument. note we eliminate MPFI
#from CT framework but not DT framework.
def time_data(N_set,DT_dt,CT_dt,runs,framework='DT',prob='KD',run_MPFI=True):
    df = pd.DataFrame(data=0,index=N_set,columns=cols_time)
    df_iter = pd.DataFrame(data=0,index=N_set,columns=cols_time)
    for i in range(runs):
        time_data, iter_data = [], []
        X, Y = {}, {}
        for N in N_set:
            print("Number of gridpoints:", N)
            d, d_iter = {}, {}
            X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
            N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
            show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
            Y[N] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
            N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
            show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
            if framework=='DT':
                d['PFI'], d_iter['PFI'] = X[N].solve_PFI('EGM',prob)[2:]
                if run_MPFI==True:
                    d['VFI'], d_iter['VFI'] = X[N].solve_MPFI('EGM',0,X[N].V0,prob)[2:]
                else:
                    d['VFI'], d_iter['VFI'] = np.inf, 0
                for relax in relax_list:
                    s = r'MPFI({0})'.format(relax)
                    d[s], d_iter[s] = X[N].solve_MPFI('EGM',relax,X[N].V0,prob)[2:]
                time_data.append(d)
                iter_data.append(d_iter)
            else:
                d['PFI'], d_iter['PFI'] = Y[N].solve_PFI()[2:]
                if run_MPFI==True:
                    d['VFI'], d_iter['VFI'] = Y[N].solve_MPFI(0)[2:]
                    for relax in relax_list:
                        s = r'MPFI({0})'.format(relax)
                        d[s], d_iter[s] = Y[N].solve_MPFI(relax)[2:]
                else:
                    d['VFI'], d_iter['VFI'] = np.inf, 0
                    for relax in relax_list:
                        s = r'MPFI({0})'.format(relax)
                        d[s], d_iter[s] = np.inf, 0
                time_data.append(d)
                iter_data.append(d_iter)
        df = df + pd.DataFrame(data=time_data,index=N_set,columns=cols_time)
        df_iter = df_iter + pd.DataFrame(data=iter_data,index=N_set,columns=cols_time)
    return df.round(decimals=n_round_time)/runs, df_iter/runs

#run times recorded by grid
def time_tables(N_set,DT_dt,CT_dt,runs,framework='DT',prob='KD'):
    N_I = N_set[0][1]
    df, df_iter = time_data(N_set,DT_dt,CT_dt,runs,framework,prob=prob)
    df.index.names, df_iter.index.names = ['Grid size'], ['Grid size']

    if framework in ['DT']:
        destin = '../../main/figures/DT_speed_stat_{0}_{1}_{2}.tex'.format(int(1000*DT_dt),prob,N_I)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

        destin = '../../main/figures/DT_iter_stat_{0}_{1}_{2}.tex'.format(int(1000*DT_dt),prob,N_I)
        with open(destin,'w') as tf:
            tf.write(df_iter.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

    if framework in ['CT']:
        destin = '../../main/figures/CT_speed_stat_{0}_{1}_{2}.tex'.format(int(10**6*CT_dt),prob,N_I)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

        destin = '../../main/figures/CT_iter_stat_{0}_{1}_{2}.tex'.format(int(10**6*CT_dt),prob,N_I)
        with open(destin,'w') as tf:
            tf.write(df_iter.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

#only EGM in the following so no need for method
def time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob='KD',norm='mean'):
    DT, CT = {}, {}
    DT['accuracy'], CT['accuracy'], DT['time'], CT['time'] = [], [], [], []

    df_acc_DT, df_acc_CT = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob=prob)
    df_DT, df_DT_iter = time_data(N_set,DT_dt,CT_dt,runs,framework='DT',prob=prob,run_MPFI=False)
    df_CT, df_CT_iter = time_data(N_set,DT_dt,CT_dt,runs,framework='CT',prob=prob,run_MPFI=False)
    df_DT = df_DT.fillna(np.inf)
    df_CT = df_CT.fillna(np.inf)

    for i in range(len(N_set)):
        if norm=='mean':
            DT['accuracy'].append(df_acc_DT.iloc[i,2])
            CT['accuracy'].append(df_acc_CT.iloc[i,2])
        else:
            DT['accuracy'].append(df_acc_DT.iloc[i,3])
            CT['accuracy'].append(df_acc_CT.iloc[i,3])
        DT['time'].append(min(df_DT.iloc[i,:]))
        CT['time'].append(min(df_CT.iloc[i,:]))
    return DT, CT

#following makes the scatterplots
def time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs,prob='KD',norm='mean'):
    DT, CT = time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob=prob,norm=norm)
    N_I = N_set[0][1]

    fig, ax = plt.subplots()
    ax.scatter(DT['accuracy'], DT['time'], marker='x',color='darkblue',lw=2,label='Discrete-time')
    ax.scatter(CT['accuracy'], CT['time'], marker='x',color='lightsteelblue',lw=2,label='Continuous-time')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylabel("Time to convergence (seconds)")
    plt.xlabel("Percent deviation from true policy function")
    plt.legend()
    #following helps ensure that y-labels are not cut off. 
    plt.tight_layout()
    if norm=='mean':
        ax.set_title('$l_1$ norm' + ' ({0} income points)'.format(N_I))
    else:
        ax.set_title('$l_{\infty}$ norm' + ' ({0} income points)'.format(N_I))
    destin = '../../main/figures/time_accuracy_{0}_{1}_{2}_{3}_{4}.eps'.format(int(10**3*DT_dt), int(10**6*CT_dt), prob, norm, N_I)
    plt.savefig(destin, format='eps', dpi=1000)
    #plt.show()
    plt.close()

"""
Create tables and scatterplots
"""

runs = 10
run_time_tables = 1

for i in range(len(parameters.income_set)):
    """
    Continuous-time and EGM accuracy
    """
    true_val = true_stat_load(parameters.DT_dt, parameters.CT_dt_true, 'KD', parameters.N_true_set[i])
    accuracy_tables(true_val, parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_true, 'both','EGM', 'KD')
    """
    Brute force
    """
    accuracy_tables(true_val, parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_true, 'DT','BF', 'KD')
    accuracy_tables(true_val, parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_true, 'DT','BF', 'Tauchen')
    """
    Sensitivity of CT methods to timestep. Only used in appendix.
    """
    accuracy_tables(true_val, parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_mid, 'CT', 'EGM', 'KD')
    accuracy_tables(true_val, parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_big, 'CT', 'EGM', 'KD')
    """
    Difference in CT and DT quantities as DT timestep decreases (only KD)
    """
    for DT_dt in [10**0, 10**-1, 10**-2]:
        comparison_tables(parameters.N_sets[i], DT_dt, parameters.CT_dt_true, 'EGM', 'KD')
    """
    Tauchen accuracy.
    """
    true_val = true_stat_load(parameters.DT_dt, parameters.CT_dt_true, 'Tauchen', parameters.N_true_set[i])
    accuracy_tables(true_val, parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_true, 'DT', 'EGM', 'Tauchen')

for i in range(len(parameters.income_set)):
    """
    Run times versus grid sizes. Only time we use "big" CT timestep.
    """
    if run_time_tables==1:
        time_tables( parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_big, runs, 'DT', 'KD')
        time_tables( parameters.N_sets[i], parameters.DT_dt, parameters.CT_dt_big, runs, 'CT', 'KD')
    """
    Run times versus accuracy scatterplot (both KD and Tauchen)
    """
    DT_data, CT_data = {}, {}
    for prob in ['KD','Tauchen']:
        true_val = true_stat_load(parameters.DT_dt, parameters.CT_dt_true, prob, parameters.N_true_set[i])
        time_accuracy_figures(true_val, parameters.N_sets_scatter[i], parameters.DT_dt, parameters.CT_dt_true, runs, prob, 'mean')
        time_accuracy_figures(true_val, parameters.N_sets_scatter[i], parameters.DT_dt, parameters.CT_dt_true, runs, prob, 'sup')
