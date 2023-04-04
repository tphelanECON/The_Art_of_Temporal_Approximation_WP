"""
Create "true" values in nonstationary setting (if they do not exist).

Only want one discrete-time true value. 
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
import time, classes, parameters
if not os.path.exists('../../main/true_values'):
    os.makedirs('../../main/true_values')

c1, c2 = parameters.c1, parameters.c2
colorFader = parameters.colorFader

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
CT_dt_mid = parameters.CT_dt_mid
CT_dt_big = parameters.CT_dt_big
DT_dt = parameters.DT_dt

def true_DT_nonstat(DT_dt,prob,N_true):
    destin = '../../main/true_values/true_V_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
    if os.path.exists(destin):
        print("Value function for nonstationary {0} framework and timestep {1} with {2} transitions already exists".format('DT',DT_dt,prob))
    else:
        print("Value function for nonstationary {0} framework and timestep {1} with {2} transitions does not exist".format('DT',DT_dt,prob))
        print("Building it now")
        X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N_true,NA=NA,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        print("Now solving for {0} gridpoints".format(N_true))
        V, c = X.nonstat_solve('BF',prob)[0:2]
        V, c = V.reshape(V.shape[0], -1), c.reshape(c.shape[0], -1)
        destin = '../../main/true_values/true_V_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/true_values/true_c_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
        pd.DataFrame(c).to_csv(destin, index=False)

def true_CT_nonstat(CT_dt,NA,N_true):
    destin_V = '../../main/true_values/true_V_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('CT',int(10**6*CT_dt),NA,N_true[0],N_true[1])
    destin_c = '../../main/true_values/true_V_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('CT',int(10**6*CT_dt),NA,N_true[0],N_true[1])
    if os.path.exists(destin_V):
        print("Value function for nonstationary {0} framework and timestep {1} and {2} ages already exists".format('CT', CT_dt, NA))
    else:
        print("Value function for nonstationary {0} framework and timestep {1} and {2} ages does not exist".format('CT', CT_dt, NA))
        print("Building it now")
        Z = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        bnd=bnd_NS,N=(N_true[0],N_true[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        V, c = Z.solve_seq_imp()[0:2]
        V, c = V.reshape(V.shape[0], -1), c.reshape(c.shape[0], -1)
        destin_V = '../../main/true_values/true_V_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('CT',int(10**6*CT_dt),NA, N_true[0],N_true[1])
        pd.DataFrame(V).to_csv(destin_V, index=False)
        destin_c = '../../main/true_values/true_c_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('CT',int(10**6*CT_dt),NA, N_true[0],N_true[1])
        pd.DataFrame(c).to_csv(destin_c, index=False)

"""
Now make the true discrete-time and continuous-time quantities.

Only want the case with 15 income points and KD transition probabilities.
"""

true_CT_nonstat(parameters.CT_dt_true,parameters.NA,parameters.N_true_set[1])
true_DT_nonstat(parameters.DT_dt,'KD',parameters.N_true_set[1])

def true_nonstat_load(DT_dt,CT_dt,CT_NA,prob,N_true):
    true_val = {}
    destin_V = '../../main/true_values/true_V_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
    destin_c = '../../main/true_values/true_c_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
    if os.path.exists(destin_V) and os.path.exists(destin_c):
        V = pd.read_csv(destin_V)
        V = np.array(V).reshape((N_true[0]+1,N_true[1]+1,parameters.NA+1))
        c = pd.read_csv(destin_c)
        c = np.array(c).reshape((N_true[0]+1,N_true[1]+1,parameters.NA+1))
        true_val['DT'] = V, c
    destin_V = '../../main/true_values/true_V_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('CT',int(10**6*CT_dt),CT_NA,N_true[0],N_true[1])
    destin_c = '../../main/true_values/true_c_{0}_nonstat_{1}_{2}_{3}_{4}.csv'.format('CT',int(10**6*CT_dt),CT_NA,N_true[0],N_true[1])
    if os.path.exists(destin_V) and os.path.exists(destin_c):
        V = pd.read_csv(destin_V)
        V = np.array(V).reshape((N_true[0]+1,N_true[1]+1,CT_NA+1))
        c = pd.read_csv(destin_c)
        c = np.array(c).reshape((N_true[0]+1,N_true[1]+1,CT_NA+1))
        true_val['CT'] = V, c
    return true_val
