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
mu, sigma = parameters.mu, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 1, 1, 1
N_true, N_c = parameters.N_true, parameters.N_c
NA_true = parameters.NA_true
NA = parameters.NA
n_round_acc = parameters.n_round_acc
n_round_time = parameters.n_round_time
CT_dt_true = parameters.CT_dt_true
CT_dt_mid = parameters.CT_dt_mid
CT_dt_big = parameters.CT_dt_big
DT_dt = parameters.DT_dt

def true_DT_nonstat(DT_dt):
    destin = '../../main/output/true_V_{0}_nonstat_{1}.csv'.format('DT', int(10**3*DT_dt))
    if os.path.exists(destin):
        print("Value function for {0} framework and timestep {1} already exists".format('DT',DT_dt))
    else:
        print("Value function for {0} framework and timestep {1} does not exist".format('DT',DT_dt))
        X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        N=N_true,NA=NA,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)
        V, c = X.nonstat_solve('BF')[0:2]
        #convert to 2-dimensional arrays.
        V, c = V.reshape(V.shape[0], -1), c.reshape(c.shape[0], -1)
        destin = '../../main/output/true_V_{0}_nonstat_{1}.csv'.format('DT', int(10**3*DT_dt))
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/output/true_c_{0}_nonstat_{1}.csv'.format('DT', int(10**3*DT_dt))
        pd.DataFrame(c).to_csv(destin, index=False)

def true_CT_nonstat(CT_dt,NA):
    destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('CT', int(10**6*CT_dt), NA)
    if os.path.exists(destin):
        print("Value function for {0} framework and timestep {1} and {2} ages already exists".format('CT', CT_dt, NA))
    else:
        print("Value function for {0} framework and timestep {1} and {2} ages already exists".format('CT', CT_dt, NA))
        Z = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        bnd=bnd_NS,N=(N_true[0],N_true[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        V, c = Z.solve_seq_imp()[0:2]
        #convert to 2-dimensional arrays.
        V, c = V.reshape(V.shape[0], -1), c.reshape(c.shape[0], -1)
        destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('CT', int(10**6*CT_dt), NA)
        pd.DataFrame(V).to_csv(destin, index=False)
        destin = '../../main/output/true_c_{0}_nonstat_{1}_{2}.csv'.format('CT', int(10**6*CT_dt), NA)
        pd.DataFrame(c).to_csv(destin, index=False)

true_CT_nonstat(CT_dt_true,parameters.NA)
true_CT_nonstat(CT_dt_true,parameters.NA_true)
true_DT_nonstat(10**0)
true_DT_nonstat(10**-1)
true_DT_nonstat(10**-2)

def true_nonstat_load(DT_dt,CT_dt,NA):
    true_val = {}
    destin = '../../main/output/true_V_{0}_nonstat_{1}.csv'.format('DT',int(10**3*DT_dt))
    if os.path.exists(destin):
        V = pd.read_csv(destin)
        V = np.array(V).reshape((N_true[0]-1,N_true[1]-1,NA-1))
    destin = '../../main/output/true_c_{0}_nonstat_{1}.csv'.format('DT',int(10**3*DT_dt))
    if os.path.exists(destin):
        c = pd.read_csv(destin)
        c = np.array(c).reshape((N_true[0]-1,N_true[1]-1,NA-1))
    true_val['DT'] = V, c
    destin = '../../main/output/true_V_{0}_nonstat_{1}_{2}.csv'.format('CT', int(10**6*CT_dt), NA)
    if os.path.exists(destin):
        V = pd.read_csv(destin)
        V = np.array(V).reshape((N_true[0]-1,N_true[1]-1,NA-1))
    destin = '../../main/output/true_c_{0}_nonstat_{1}_{2}.csv'.format('CT', int(10**6*CT_dt), NA)
    if os.path.exists(destin):
        c = pd.read_csv(destin)
        c = np.array(c).reshape((N_true[0]-1,N_true[1]-1,NA-1))
    true_val['CT'] = V, c
    return true_val
