"""
Create all of the "true" values in the stationary setting (if they do not exist).
At the moment I think we need five of these:
    * one for the continuous-time case, using dt =10**-6.
    * three for the discrete-time case with KD transitions, using dt= 1, 0.1,
    and 0.01, respectively.
    * one for the discrete-time case with Tauchen transitions, using dt = 1.

For the discrete-time Tauchen we use MPFI not PFI because the transition matrix
is much less sparse and convergence would take a very long time.
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
import time, classes, parameters
if not os.path.exists('../../main/output'):
    os.makedirs('../../main/output')
from scipy.interpolate import interp1d

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

def true_DT_stat(DT_dt,prob):
    destin = '../../main/output/true_V_{0}_stat_{1}_{2}.csv'.format('DT',int(10**3*DT_dt),prob)
    if os.path.exists(destin):
        print("Value function for stationary {0} framework and timestep {1} with {2} transitions already exists".format('DT',DT_dt,prob))
    else:
        print("Value function for stationary {0} framework and timestep {1} with {2} transitions does not exist".format('DT',DT_dt,prob))
        X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N_true,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        print("Now solving for {0} gridpoints".format(N_true))
        if prob=='KD':
            V, c = X.solve_PFI(method='BF',prob=prob)[0:2]
        else:
            V_init = X.V(X.c0,'KD')
            V, c = X.solve_MPFI(method='BF',M=100,V_init=V_init,prob=prob)[0:2]
        destin = '../../main/output/true_V_{0}_stat_{1}_{2}.csv'.format('DT',int(10**3*DT_dt),prob)
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/output/true_c_{0}_stat_{1}_{2}.csv'.format('DT',int(10**3*DT_dt),prob)
        pd.DataFrame(c).to_csv(destin, index=False)

def true_CT_stat(CT_dt):
    destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('CT',int(10**6*CT_dt))
    if os.path.exists(destin):
        print("Value function for stationary {0} framework and timestep {1} already exists".format('CT',CT_dt))
    else:
        print("Value function for stationary {0} framework and timestep {1} does not exist".format('CT',CT_dt))
        Y = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N_true,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        V, c = Y.solve_PFI()[0:2]
        destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('CT',int(10**6*CT_dt))
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/output/true_c_{0}_stat_{1}.csv'.format('CT',int(10**6*CT_dt))
        pd.DataFrame(c).to_csv(destin, index=False)

"""
Now make the true discrete-time and continuous-time quantities
"""

true_CT_stat(CT_dt_true)
true_DT_stat(10**0,'KD')
true_DT_stat(10**-1,'KD')
true_DT_stat(10**-2,'KD')
true_DT_stat(10**0,'Tauchen')

def true_stat_load(DT_dt,CT_dt,prob):
    true_val = {}
    destin = '../../main/output/true_V_{0}_stat_{1}_{2}.csv'.format('DT',int(10**3*DT_dt),prob)
    if os.path.exists(destin):
        V = pd.read_csv(destin)
    destin = '../../main/output/true_c_{0}_stat_{1}_{2}.csv'.format('DT',int(10**3*DT_dt),prob)
    if os.path.exists(destin):
        c = pd.read_csv(destin)
    true_val['DT'] = np.array(V), np.array(c)
    destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('CT',int(10**6*CT_dt))
    if os.path.exists(destin):
        V = pd.read_csv(destin)
    destin = '../../main/output/true_c_{0}_stat_{1}.csv'.format('CT',int(10**6*CT_dt))
    if os.path.exists(destin):
        c = pd.read_csv(destin)
    true_val['CT'] = np.array(V), np.array(c)
    return true_val
