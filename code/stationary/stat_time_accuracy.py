"""
Accuracy and run times for stationary income fluctuation problems.

Relevant class constructors imported from classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent).
    CT_stat_IFP: continuous-time stationary IFP.

All solve functions (solve_MPFI, solve_PFI) in above class constructors
return quadruples (V, c, toc-tic, i) where i = no. of iterations.

Methods:

    * accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF',prob='KD'):
    takes dictionary true_val of true values, list of grid sizes N_set for
    assets and income, DT time step DT_dt and CT time step CT_dt and returns
    either one or three dataframes, indicating distance between computed and "true"
    values, or difference between the computed quantities across DT and CT.
    For DT also specify policy updating method, 'BF' or 'EGM'.
    * accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF',prob='KD'):
    produces tables using accuracy_data.
    * time_data(N_set,DT_dt,CT_dt,runs,framework,prob,run_MPFI=True): computes
    run times to solve problems on grids given by N_set, for DT and CT. Averages
    over 'runs' number of runs. run_MPFI=False avoids performing ANY of the MPFI.
    * time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob): creates tables using time_data.
    * time_tables(N_set,DT_dt,CT_dt,runs,framework='DT',prob='KD'): creates
    tables for the time taken to solve problems on various grids.
    * time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs,prob='KD'): creates the scatterplot.

true_stat(DT_dt,CT_dt,prob) gets "true" discrete-time and continuous-time value
functions and policy functions. Returns dictionary with keys ['DT', 'CT'] and for
each returns V, c tuple. This function is called here but defined in true_stat.py.

Use parameters.N_set for the tables.
Use parameters.N_set_scatter for the scatterplot.

true_stat_load now requires dimensions.

There does not seem to be an need to compute the "true" values for a small dt
in the DT case. We only did this to compare across frameworks. The comparison
between DT and CT does not require knowledge of the true values.

Consider breaking this off? Yes. Doing this now. No longer called "accuracy".
Now called comparison_data and comparison_tables. The comparison ones do not
need a "true" quantity and so this should be much quicker.


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
Compare with true values (accuracy_data no longer contains the comparisons across methods).
Want to rewrite following in percentage deviations of policy function.
"""

def accuracy_data(true_val,N_set,DT_dt,CT_dt,framework='both',method='EGM',prob='KD'):
    N_true_shape = true_val['DT'][0].shape
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
        return pd.DataFrame(data=data_DT,index=N_set,columns=cols_true), \
        pd.DataFrame(data=data_CT,index=N_set,columns=cols_true)

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

def accuracy_tables(true_val,N_set,DT_dt,CT_dt,framework='both',method='BF',prob='KD'):
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
        destin = '../../main/figures/DT_{0}_accuracy_stat_{1}_{2}.tex'.format(method,int(10**3*DT_dt),prob)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

    if framework in ['CT','both']:
        df = pd.DataFrame(data=df_CT,index=N_set,columns=cols_true)
        df = df[cols_true].round(decimals=n_round_acc)
        df.index.names = ['Grid size']
        destin = '../../main/figures/CT_accuracy_stat_{0}_{1}.tex'.format(int(10**6*CT_dt),prob)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_true)+1)))

def comparison_tables(N_set,DT_dt,CT_dt,method='EGM',prob='KD'):
    df_compare = comparison_data(N_set,DT_dt,CT_dt,method,prob)

    df = pd.DataFrame(data=df_compare,index=N_set,columns=cols_compare)
    df = df[cols_compare].round(decimals=n_round_acc)
    df.index.names = ['Grid size']

    destin = '../../main/figures/DT_CT_{0}_accuracy_stat_{1}_{2}_{3}.tex'.format(method,int(10**3*DT_dt),int(10**6*CT_dt),prob)
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_compare)+1)))

#following only uses EGM so no need for method argument
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
                    for relax in relax_list:
                        s = r'MPFI ({0})'.format(relax)
                        d[s], d_iter[s] = X[N].solve_MPFI('EGM',relax,X[N].V0,prob)[2:]
                else:
                    d['VFI'], d_iter['VFI'] = np.inf, 0
                    for relax in relax_list:
                        s = r'MPFI ({0})'.format(relax)
                        d[s], d_iter[s] = np.inf, 0
                time_data.append(d)
                iter_data.append(d_iter)
            else:
                d['PFI'], d_iter['PFI'] = Y[N].solve_PFI()[2:]
                if run_MPFI==True:
                    d['VFI'], d_iter['VFI'] = Y[N].solve_MPFI(0)[2:]
                    for relax in relax_list:
                        s = r'MPFI ({0})'.format(relax)
                        d[s], d_iter[s] = Y[N].solve_MPFI(relax)[2:]
                else:
                    d['VFI'], d_iter['VFI'] = np.inf, 0
                    for relax in relax_list:
                        s = r'MPFI ({0})'.format(relax)
                        d[s], d_iter[s] = np.inf, 0
                time_data.append(d)
                iter_data.append(d_iter)
        df = df + pd.DataFrame(data=time_data,index=N_set,columns=cols_time)
        df_iter = df_iter + pd.DataFrame(data=iter_data,index=N_set,columns=cols_time)
    return df.round(decimals=n_round_time)/runs, df_iter/runs

def time_tables(N_set,DT_dt,CT_dt,runs,framework='DT',prob='KD'):
    df, df_iter = time_data(N_set,DT_dt,CT_dt,runs,framework,prob=prob)
    df.index.names, df_iter.index.names = ['Grid size'], ['Grid size']

    if framework in ['DT']:
        destin = '../../main/figures/DT_speed_stat_{0}_{1}.tex'.format(int(1000*DT_dt),prob)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

        destin = '../../main/figures/DT_iter_stat_{0}_{1}.tex'.format(int(1000*DT_dt),prob)
        with open(destin,'w') as tf:
            tf.write(df_iter.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

    if framework in ['CT']:
        destin = '../../main/figures/CT_speed_stat_{0}_{1}.tex'.format(int(10**6*CT_dt),prob)
        with open(destin,'w') as tf:
            tf.write(df.to_latex(escape=False,column_format='c'*(len(cols_time)+1)))

        destin = '../../main/figures/CT_iter_stat_{0}_{1}.tex'.format(int(10**6*CT_dt),prob)
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
        #DT['time'].append(min(df_DT.iloc[i,:]))
        DT['time'].append(min(df_DT.iloc[i,:]))
        CT['time'].append(min(df_CT.iloc[i,:]))
        #CT['time'].append(min(df_CT.iloc[i,:]))
    return DT, CT

def time_accuracy_figures(true_val,N_set,DT_dt,CT_dt,runs,prob='KD',norm='mean'):
    DT, CT = time_accuracy_data(true_val,N_set,DT_dt,CT_dt,runs,prob=prob,norm=norm)

    fig, ax = plt.subplots()
    ax.scatter(DT['accuracy'], DT['time'], marker='x',color='darkblue',lw=2,label='Discrete-time')
    ax.scatter(CT['accuracy'], CT['time'], marker='x',color='lightsteelblue',lw=2,label='Continuous-time')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.ylabel("Time to convergence (seconds)")
    plt.xlabel("Percent deviation from true policy function")
    plt.legend()
    if norm=='mean':
        ax.set_title('Speed versus accuracy ($l_1$ norm)')
    else:
        ax.set_title('Speed versus accuracy ($l_{\infty}$ norm)')
    destin = '../../main/figures/time_accuracy_{0}_{1}_{2}_{3}.eps'.format(int(10**3*DT_dt), int(10**6*CT_dt), prob, norm)
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

"""
Create tables (to avoid confusion don't drop parameters prefix)
"""

"""
Accuracy of CT and DT
"""
true_val = true_stat_load(parameters.DT_dt, parameters.CT_dt_true, 'KD', parameters.N_true)
accuracy_tables(true_val, parameters.N_set, parameters.DT_dt, parameters.CT_dt_true, 'both','EGM', 'KD')
"""
Sensitivity of continuous-time methods to timestep (just CT here)
"""
accuracy_tables(true_val, parameters.N_set, parameters.DT_dt, parameters.CT_dt_mid, 'CT', 'EGM', 'KD')
accuracy_tables(true_val, parameters.N_set, parameters.DT_dt, parameters.CT_dt_big, 'CT', 'EGM', 'KD')
"""
Difference in CT and DT quantities as DT timestep decreases (only KD)
"""
for DT_dt in [10**0, 10**-1, 10**-2]:
    comparison_tables(parameters.N_set, DT_dt, parameters.CT_dt_true, 'EGM', 'KD')
"""
Tauchen accuracy
"""
true_val = true_stat_load(parameters.DT_dt, parameters.CT_dt_true, 'Tauchen', parameters.N_true)
accuracy_tables(true_val, parameters.N_set, parameters.DT_dt, parameters.CT_dt_true, 'DT', 'EGM', 'Tauchen')
"""
Time
"""
runs=20
run_time_tables=1

"""
Run times versus grid sizes (tables)
"""
if run_time_tables==1:
    time_tables( parameters.N_set, parameters.DT_dt, parameters.CT_dt_big, runs, 'DT', 'KD')
    time_tables( parameters.N_set, parameters.DT_dt, parameters.CT_dt_big, runs, 'CT', 'KD')
"""
Run times versus accuracy scatterplot (both KD and Tauchen)
"""
DT_data, CT_data = {}, {}
for prob in ['KD','Tauchen']:
    true_val = true_stat_load(parameters.DT_dt, parameters.CT_dt_true, prob, parameters.N_true)
    DT_data[prob], CT_data[prob] = time_accuracy_data(true_val,parameters.N_set_scatter, parameters.DT_dt, parameters.CT_dt_true, runs ,prob,'mean')
    time_accuracy_figures(true_val, parameters.N_set_scatter, parameters.DT_dt, parameters.CT_dt_true, runs, prob, 'mean')
    time_accuracy_figures(true_val, parameters.N_set_scatter, parameters.DT_dt, parameters.CT_dt_true, runs, prob, 'sup')
