"""
Create all of the "true" values in the stationary setting (if they do not exist).

We need three of these (possibly more):
    * One for the CT case, using the "small" dt = 10**-6.
    * One for the DT case with KD transitions, using dt= 1,
    * One for the DT case with Tauchen transitions, using dt = 1.

No DT quantities for dt=0.1, 0.01. Does not add anything beyond comparison on coarse grids.

true_stat(DT_dt,CT_dt,prob) gets "true" discrete-time and continuous-time value
and policy functions. Returns dict with keys ['DT', 'CT']; for each returns V, c.
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import time, classes, parameters
from scipy.interpolate import interp1d
"""
Make folder in which to place true values if it does not already exist
"""
if not os.path.exists('../../main/true_values'):
    os.makedirs('../../main/true_values')
"""
Preference and income parameters
"""
rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
"""
Grid and numerical parameters
"""
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
show_iter, show_method, show_final = 1, 1, 1
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

N_t = parameters.N_t
N_c = parameters.N_c
"""
Time steps used.
"""
"""
Function creating true discrete-time quantities. Arguments:
    * timestep DT_dt
    * transition probabilities "prob" (KD or Tauchen)
    * size of "true" quantities
"""
def true_DT_stat(DT_dt,prob,N_true):
    destin = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
    if os.path.exists(destin):
        print("Value function for stationary {0} framework and timestep {1} with {2} transitions already exists".format('DT',DT_dt,prob))
    else:
        print("Value function for stationary {0} framework and timestep {1} with {2} transitions does not exist".format('DT',DT_dt,prob))
        print("Building it now")
        X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N_true,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        print("Now solving for {0} gridpoints".format(N_true))
        #V, c = X.solve_MPFI('EGM',0,X.V0,prob)[0:2]
        V, c = X.solve_PFI(method='BF',prob=prob)[0:2]
        destin = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/true_values/true_c_{0}_stat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
        pd.DataFrame(c).to_csv(destin, index=False)

def true_CT_stat(CT_dt,N_true):
    destin = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}.csv'.format('CT',int(10**6*CT_dt),N_true[0],N_true[1])
    if os.path.exists(destin):
        print("Value function for stationary {0} framework and timestep {1} already exists".format('CT',CT_dt))
    else:
        print("Value function for stationary {0} framework and timestep {1} does not exist".format('CT',CT_dt))
        print("Building it now")
        Y = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N_true,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        V, c = Y.solve_PFI()[0:2]
        destin = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}.csv'.format('CT',int(10**6*CT_dt),N_true[0],N_true[1])
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/true_values/true_c_{0}_stat_{1}_{2}_{3}.csv'.format('CT',int(10**6*CT_dt),N_true[0],N_true[1])
        pd.DataFrame(c).to_csv(destin, index=False)
"""
Build true DT and CT quantities if they do not exist
"""
for i in range(1,3):
#for i in range(len(parameters.income_set)):
    true_CT_stat(parameters.CT_dt_true,parameters.N_true_set[i])
    true_DT_stat(parameters.DT_dt,'KD',parameters.N_true_set[i])
    true_DT_stat(parameters.DT_dt,'Tauchen',parameters.N_true_set[i])
"""
Following loads true quantities, showing an error if none exist.
"""
def true_stat_load(DT_dt,CT_dt,prob,N_true):
    true_val = {}
    destin_V = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
    destin_c = '../../main/true_values/true_c_{0}_stat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
    if os.path.exists(destin_V) and os.path.exists(destin_c):
        V = pd.read_csv(destin_V)
        c = pd.read_csv(destin_c)
        true_val['DT'] = np.array(V), np.array(c)
    destin_V = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}.csv'.format('CT',int(10**6*CT_dt),N_true[0],N_true[1])
    destin_c = '../../main/true_values/true_c_{0}_stat_{1}_{2}_{3}.csv'.format('CT',int(10**6*CT_dt),N_true[0],N_true[1])
    if os.path.exists(destin_V) and os.path.exists(destin_c):
        V = pd.read_csv(destin_V)
        c = pd.read_csv(destin_c)
        true_val['CT'] = np.array(V), np.array(c)
    return true_val
