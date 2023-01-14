"""
This script computes accuracy and run times for stationary income fluctuation problems.

Relevant class constructors imported from classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent).
    CT_stat_IFP: continuous-time stationary IFP.

Comparison for brute force method with the continuous-time method, both between
each other and with respect to the "true" values.

All solve functions (solve_MPFI, solve_PFI) in above class constructors
return quadruples (V, c, toc-tic, i) where i = no. of iterations.

functions:

    * true_stat(DT_dt,CT_dt): gets the "true" discrete-time and continuous-time
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

True quantities created in create_true_stat.py. I will use this as a template
for everything else.
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
if not os.path.exists('../../main/output'):
    os.makedirs('../../main/output')
from true_stat import true_stat_load

c1, c2 = parameters.c1, parameters.c2
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
First compare discrete-time and continuous-time quantities with their respective
"true" counterparts (computed on fine asset grid).
"""

cols_true = ['$||c - c_{\textnormal{true}}||_1$',
'$||c - c_{\textnormal{true}}||_{\infty}$',
'$||V - V_{\textnormal{true}}||_1$',
'$||V - V_{\textnormal{true}}||_{\infty}$']

cols_compare = ['$||c_{DT} - c_{CT}||_1$',
'$||c_{DT} - c_{CT}||_{\infty}$',
'$||V_{DT} - V_{CT}||_1$',
'$||V_{DT} - V_{CT}||_{\infty}$']

relax_list = parameters.relax_list
cols_time = ['VFI']
for relax in relax_list:
    cols_time.append(r'MPFI ({0})'.format(relax))
cols_time.append('PFI')

"""
Get the true values or state that they are not in memory. Following does not
COMPUTE anything. It just gets the data or tells us if it is missing.
"""

X_true = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
N=N_true,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)

Y_true = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
N=N_true,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt_true)

"""
Compare with true values
"""

def accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF'):
    X, Y, DT, CT = {}, {}, {}, {}
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
        N=N,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        Y[N] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        def compare(f1,f2):
            f2_fine = np.zeros((N_true[0]-1,N_true[1]-1))
            for j in range(N_true[1]-1):
                f2_fine[:,j] = interp1d(X[N].grid[0], f2[:,j],fill_value="extrapolate")(X_true.grid[0])
            return f1-f2_fine
        #compute DT output and compare with truth
        if framework in ['DT','both']:
            DT[N] = X[N].solve_PFI(method)
            d = np.array(compare(DT['True'][0], DT[N][0])), np.array(compare(DT['True'][1], DT[N][1]))
            d_DT[cols_true[0]] = np.mean(np.abs(d[1]))
            d_DT[cols_true[1]] = np.max(np.abs(d[1]))
            d_DT[cols_true[2]] = np.mean(np.abs(d[0]))
            d_DT[cols_true[3]] = np.max(np.abs(d[0]))
            data_DT.append(d_DT)
        #compute CT output and compare with truth
        if framework in ['CT','both']:
            CT[N] = Y[N].solve_PFI()
            d = np.array(compare(CT['True'][0], CT[N][0])), np.array(compare(CT['True'][1], CT[N][1]))
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

def accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF'):
    if framework=='DT':
        df_data = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method)
        df = pd.DataFrame(data=df_data,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/DT_accuracy_stat_{0}.tex'.format(int(10**3*DT_dt))
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    elif framework=='CT':
        df_data = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method)
        df = pd.DataFrame(data=df_data,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/CT_accuracy_stat_{0}.tex'.format(int(10**6*CT_dt))
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    elif framework=='both':
        df_DT, df_CT, df_compare = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework,method)
        df = pd.DataFrame(data=df_compare,index=N_set,columns=cols_compare)
        df = df[cols_compare].round(decimals=n_round_acc)
        df.index.names = ['Grid size']

        destin = '../../main/figures/DT_CT_accuracy_stat_{0}_{1}.tex'.format(int(10**3*DT_dt), int(10**6*CT_dt))
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_compare)+1)))

def time_data(N_set,DT_dt,CT_dt,runs,framework='DT',suppress_VFI=False):
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
                d['PFI'], d_iter['PFI'] = X[N].solve_PFI('EGM')[2:]
                if suppress_VFI==False:
                    d['VFI'], d_iter['VFI'] = X[N].solve_MPFI('EGM',0,X[N].V0)[2:]
                else:
                    d['VFI'], d_iter['VFI'] = np.inf, 0
                for relax in relax_list:
                    s = r'MPFI ({0})'.format(relax)
                    d[s], d_iter[s] = X[N].solve_MPFI('EGM',relax,X[N].V0)[2:]
                time_data.append(d)
                iter_data.append(d_iter)
            else:
                d['PFI'], d_iter['PFI'] = Y[N].solve_PFI()[2:]
                if suppress_VFI==False:
                    d['VFI'], d_iter['VFI'] = Y[N].solve_MPFI(0)[2:]
                else:
                    d['VFI'], d_iter['VFI'] = np.inf, 0
                for relax in relax_list:
                    s = r'MPFI ({0})'.format(relax)
                    d[s], d_iter[s] = Y[N].solve_MPFI(relax)[2:]
                time_data.append(d)
                iter_data.append(d_iter)
        df = df + pd.DataFrame(data=time_data,index=N_set,columns=cols_time)
        df_iter = df_iter + pd.DataFrame(data=iter_data,index=N_set,columns=cols_time)
    return df.round(decimals=n_round_time)/runs, df_iter

def time_tables(true_val,N_set,DT_dt,CT_dt,runs,framework='DT'):
    df, df_iter = time_data(N_set,DT_dt,CT_dt,runs,framework)
    df.index.names, df_iter.index.names = ['Grid size'], ['Grid size']

    if framework=='DT':
        destin = '../../main/figures/DT_speed_stat_{0}.tex'.format(int(1000*DT_dt))
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

        destin = '../../main/figures/DT_iter_stat_{0}.tex'.format(int(1000*DT_dt))
        with open(destin,'w') as tf:
            tf.write(df_iter.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

    if framework=='CT':
        destin = '../../main/figures/CT_speed_stat_{0}.tex'.format(int(1000*CT_dt))
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

        destin = '../../main/figures/CT_iter_stat_{0}.tex'.format(int(1000*CT_dt))
        with open(destin,'w') as tf:
            tf.write(df_iter.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))
"""
Use EGM for the accuracy in the following, because that is what is used in the time tables.
"""
def time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs):
    DT, CT = {}, {}
    DT['accuracy'], CT['accuracy'] = [], []
    DT['time'], CT['time'] = [], []

    df_acc_DT, df_acc_CT, df_acc_compare = accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM')
    df_DT, df_DT_iter = time_data(N_set,DT_dt,CT_dt,runs,framework='DT',suppress_VFI=True)
    df_CT, df_CT_iter = time_data(N_set,DT_dt,CT_dt,runs,framework='CT',suppress_VFI=True)

    df_DT = df_DT.fillna(np.inf)
    df_CT = df_CT.fillna(np.inf)

    for i in range(len(N_set)):
        DT['accuracy'].append(df_acc_DT.iloc[i,0])
        CT['accuracy'].append(df_acc_CT.iloc[i,0])
        DT['time'].append(min(df_DT.iloc[i,:]))
        CT['time'].append(min(df_CT.iloc[i,:]))
    return DT, CT

def time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs):
    DT, CT = time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs)

    fig, ax = plt.subplots()
    ax.scatter(DT['accuracy'], DT['time'], marker='x',color='darkblue',lw=2,label='Discrete-time')
    ax.scatter(CT['accuracy'], CT['time'], marker='x',color='lightsteelblue',lw=2,label='Continuous-time')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylabel("Time to convergence (seconds)")
    plt.xlabel("Accuracy")
    plt.legend()
    ax.set_title('Speed versus accuracy tradeoff')
    destin = '../../main/figures/time_accuracy_{0}_{1}.eps'.format(int(10**3*DT_dt), int(10**6*CT_dt))
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

"""
Create tables
"""

runs=1
N_grid = np.linspace(np.log(50), np.log(1000), 10)
N_set_big = [(int(np.rint(np.exp(n))),10) for n in N_grid]

DT_dt = 10**0
true_val = true_stat_load(DT_dt, CT_dt_true)
accuracy_tables(true_val, parameters.N_set, DT_dt, CT_dt_true, 'both')
accuracy_tables(true_val, parameters.N_set, DT_dt, CT_dt_mid, 'CT')
accuracy_tables(true_val, parameters.N_set, DT_dt, CT_dt_big, 'CT')
DT_dt = 10**-1
true_val = true_stat_load(DT_dt, CT_dt_true)
accuracy_tables(true_val, parameters.N_set, DT_dt, CT_dt_true, 'both')
DT_dt = 10**-2
true_val = true_stat_load(DT_dt, CT_dt_true)
accuracy_tables(true_val, parameters.N_set, DT_dt, CT_dt_true, 'both')
DT_dt = 10**0
true_val = true_stat_load(DT_dt, CT_dt_true)
time_tables(true_val, parameters.N_set, DT_dt, CT_dt_big, runs, 'DT')
time_tables(true_val, parameters.N_set, DT_dt, CT_dt_big, runs, 'CT')
time_accuracy_figures(true_val, N_set_big, DT_dt, CT_dt_big, runs)
