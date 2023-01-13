"""
Create all of the "true" values in the stationary setting (if they do not exist).

One "true" set of quantities for the continuous-time case, and three for the
discrete-time case.
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

c1, c2 = parameters.c1, parameters.c2
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

def true_DT_stat(DT_dt):
    destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('DT', int(10**3*DT_dt))
    if os.path.exists(destin):
        print("Value function for {0} framework and timestep {1} already exists".format('DT',DT_dt))
    else:
        print("Value function for {0} framework and timestep {1} does not exist".format('DT',DT_dt))
        X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        N=N_true,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        V, c = X.solve_PFI('BF')[0:2]
        destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('DT', int(10**3*DT_dt))
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/output/true_c_{0}_stat_{1}.csv'.format('DT', int(10**3*DT_dt))
        pd.DataFrame(c).to_csv(destin, index=False)

def true_CT_stat(CT_dt):
    destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('CT', int(10**6*CT_dt))
    if os.path.exists(destin):
        print("Value function for {0} framework and timestep {1} already exists".format('CT', CT_dt))
    else:
        print("Value function for {0} framework and timestep {1} does not exist".format('CT', CT_dt))
        Y = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        N=N_true,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        V, c = Y.solve_PFI()[0:2]
        destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('CT', int(10**6*CT_dt))
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/output/true_c_{0}_stat_{1}.csv'.format('CT', int(10**6*CT_dt))
        pd.DataFrame(c).to_csv(destin, index=False)

"""
Now make the true discrete-time and continuous-time quantities
"""

true_DT_stat(10**0)
true_DT_stat(10**-1)
true_DT_stat(10**-2)
true_CT_stat(CT_dt_true)

def true_stat_load(DT_dt,CT_dt):
    true_val = {}
    destin = '../../main/output/true_V_{0}_stat_{1}.csv'.format('DT',int(10**3*DT_dt))
    if os.path.exists(destin):
        V = pd.read_csv(destin)
    destin = '../../main/output/true_c_{0}_stat_{1}.csv'.format('DT',int(10**3*DT_dt))
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
