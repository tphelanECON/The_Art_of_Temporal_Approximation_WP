"""
Create all of the "true" values in the stationary setting (if they do not exist).

We need five of these (possibly more):
    * One for the continuous-time case, using the "small" dt = 10**-6.
    * Three for the discrete-time case with KD transitions, using dt= 1, 0.1,
    and 0.01, resp., to document convergence of CT and DT quantities to one another.
    * One for the discrete-time case with Tauchen transitions, using dt = 1.

Recall that the KD transitions in the discrete-time case with timestep dt are
always calculated by constructing the transition probabilites for some fixed
choice of small dt used in the continuous-time case and then sampling this at a
lower frequency.
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
CT_dt_true = parameters.CT_dt_true
DT_dt = parameters.DT_dt
"""
Function creating true discrete-time quantities.
Arguments: timestep DT_dt and transition probabilities "prob" (KD or Tauchen).
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
Now build the true discrete-time and continuous-time quantities if they do not exist
"""

true_CT_stat(CT_dt_true,parameters.N_true)
true_DT_stat(10**0,'KD',parameters.N_true)
true_DT_stat(10**-1,'KD',parameters.N_true)
true_DT_stat(10**-2,'KD',parameters.N_true)
true_DT_stat(10**0,'Tauchen',parameters.N_true)

"""
Following loads the true quantities. It will throw an error if none exist.
"""
def true_stat_load(DT_dt,CT_dt,prob,N_true):
    true_val = {}
    destin = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
    if os.path.exists(destin):
        V = pd.read_csv(destin)
    destin = '../../main/true_values/true_c_{0}_stat_{1}_{2}_{3}_{4}.csv'.format('DT',int(10**3*DT_dt),prob,N_true[0],N_true[1])
    if os.path.exists(destin):
        c = pd.read_csv(destin)
    true_val['DT'] = np.array(V), np.array(c)
    destin = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}.csv'.format('CT',int(10**6*CT_dt),N_true[0],N_true[1])
    if os.path.exists(destin):
        V = pd.read_csv(destin)
    destin = '../../main/true_values/true_V_{0}_stat_{1}_{2}_{3}.csv'.format('CT',int(10**6*CT_dt),N_true[0],N_true[1])
    if os.path.exists(destin):
        c = pd.read_csv(destin)
    true_val['CT'] = np.array(V), np.array(c)
    return true_val
