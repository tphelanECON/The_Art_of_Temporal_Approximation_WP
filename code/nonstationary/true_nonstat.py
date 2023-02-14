"""
Create all of the "true" values in the stationary setting (if they do not exist).

Stationary:
    * one quantities for continuous-time case (CT_dt = 10**-6);
    * three for the discrete-time case (DT_dt = 1, 10**-1, 10**-2).

Nonstationary:
    * two "true" set of quantities for continuous-time case (CT_dt = 10**-6, NA = NA, NA = NA_true);
    * one for the discrete-time case (DT_dt = 1, NA = NA).

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
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 1, 1, 1
N_true = parameters.N_true
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

def true_DT_nonstat(DT_dt,prob):
    destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('DT',int(10**3*DT_dt),prob)
    if os.path.exists(destin):
        print("Value function for nonstationary {0} framework and timestep {1} with {2} transitions already exists".format('DT',DT_dt,prob))
    else:
        print("Value function for nonstationary {0} framework and timestep {1} with {2} transitions does not exist".format('DT',DT_dt,prob))
        X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        N=N_true,NA=NA,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        print("Now solving for {0} gridpoints".format(N_true))
        V, c = X.nonstat_solve('BF',prob)[0:2]
        V, c = V.reshape(V.shape[0], -1), c.reshape(c.shape[0], -1)
        destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('DT', int(10**3*DT_dt), prob)
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/output/true_c_{0}_nonstat_{1}_{2}.csv'.format('DT', int(10**3*DT_dt), prob)
        pd.DataFrame(c).to_csv(destin, index=False)

def true_CT_nonstat(CT_dt,NA):
    destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('CT', int(10**6*CT_dt), NA)
    if os.path.exists(destin):
        print("Value function for nonstationary {0} framework and timestep {1} and {2} ages already exists".format('CT', CT_dt, NA))
    else:
        print("Value function for nonstationary {0} framework and timestep {1} and {2} ages does not exist".format('CT', CT_dt, NA))
        Z = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        bnd=bnd_NS,N=(N_true[0],N_true[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        V, c = Z.solve_seq_imp()[0:2]
        V, c = V.reshape(V.shape[0], -1), c.reshape(c.shape[0], -1)
        destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('CT', int(10**6*CT_dt), NA)
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/output/true_c_{0}_nonstat_{1}_{2}.csv'.format('CT', int(10**6*CT_dt), NA)
        pd.DataFrame(c).to_csv(destin, index=False)

"""
Now make the true discrete-time and continuous-time quantities
"""

true_CT_nonstat(CT_dt_true,parameters.NA)
true_CT_nonstat(CT_dt_true,parameters.NA_true)
true_DT_nonstat(10**0,'KD')
true_DT_nonstat(10**0,'Tauchen')

#NA in the following will be the number of agesteps in the continuous-time case.
def true_nonstat_load(DT_dt,CT_dt,CT_NA,prob):
    true_val = {}
    destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('DT',int(10**3*DT_dt),prob)
    if os.path.exists(destin):
        V = pd.read_csv(destin)
        V = np.array(V).reshape((N_true[0]-1,N_true[1]-1,parameters.NA-1))
    destin = '../../main/output/true_c_{0}_nonstat_{1}_{2}.csv'.format('DT',int(10**3*DT_dt),prob)
    if os.path.exists(destin):
        c = pd.read_csv(destin)
        c = np.array(c).reshape((N_true[0]-1,N_true[1]-1,parameters.NA-1))
    true_val['DT'] = V, c
    destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('CT',int(10**6*CT_dt),CT_NA)
    if os.path.exists(destin):
        V = pd.read_csv(destin)
        V = np.array(V).reshape((N_true[0]-1,N_true[1]-1,CT_NA-1))
    destin = '../../main/output/true_c_{0}_nonstat_{1}_{2}.csv'.format('CT',int(10**6*CT_dt),CT_NA)
    if os.path.exists(destin):
        c = pd.read_csv(destin)
        c = np.array(c).reshape((N_true[0]-1,N_true[1]-1,CT_NA-1))
    true_val['CT'] = V, c
    return true_val

true_val = {}

DT_dt, CT_dt = 10**0, 10*-6
for prob in ['KD','Tauchen']:
    true_val['(coarse, {0})'.format(prob)] = true_nonstat_load(DT_dt,CT_dt,parameters.NA,prob)
    true_val['(fine, {0})'.format(prob)] = true_nonstat_load(DT_dt,CT_dt,parameters.NA_true,prob)
