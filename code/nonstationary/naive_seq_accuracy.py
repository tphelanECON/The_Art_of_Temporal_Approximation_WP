"""
Comparison of naive PFI and sequential PFI.

This script is simply to provide us confidence in the code and the sparse solver.
These two approaches literally solve the same system of equations and so the
difference between them ought to be minuscule.

JANUARY 22: difference is small (approx 10**-9) but not minuscule. In a previous
version of the code the difference were of the order of 10**-12 or 10**-13.

However, differences are smaller than the tolerance used, and several orders of
mangitude small than the differences between coarse and fine grids.

We have a more basic problem: we need to ensure that the naive and the sequential
PFI are
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import time, classes, parameters #, classes_old

c1,c2 = parameters.c1, parameters.c2
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
NA = parameters.NA
cols_compare = parameters.cols_compare

"""
Functions for data construction and table creation
"""

def accuracy_data(N_set,CT_dt):
    """
    Pre-allocate all quantities
    """
    Z, naive, seq, naive_seq = {}, {}, {}, {}
    data_compare = []
    for N in N_set:
        print("Number of gridpoints:", N)
        d_compare = {}
        Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
        bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        naive[N] = Z[N].solve_PFI()
        seq[N] = Z[N].solve_seq_imp()
        naive_seq[N] = naive[N][0]-seq[N][0], naive[N][1]-seq[N][1]
        #compare naive- and sequential PFI output
        d_compare[cols_compare[0]] = np.mean(np.abs(naive_seq[N][1]))
        d_compare[cols_compare[1]] = np.max(np.abs(naive_seq[N][1]))
        d_compare[cols_compare[2]] = np.mean(np.abs(naive_seq[N][0]))
        d_compare[cols_compare[3]] = np.max(np.abs(naive_seq[N][0]))
        #append the above results
        data_compare.append(d_compare)
    return data_compare

def accuracy_tables(N_set,CT_dt):
    data_compare = accuracy_data(N_set,CT_dt)

    df = pd.DataFrame(data=data_compare,index=N_set,columns=cols_compare)
    df.index.names = ['Grid size']

    destin = '../../main/figures/naive_seq_accuracy.tex'
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

"""
Create tables
"""

CT_dt = CT_dt_true
N_set = [(50,10),(100,10),(150,10)]
#accuracy_tables(N_set,CT_dt)

data_compare = accuracy_data(N_set,CT_dt)

N = (50,10)
Z = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
n = Z.solve_PFI()
s = Z.solve_seq_imp()

#seems to be some missing mass somewhere. There must be.

cdiff = n[1]- s[1]
cdiff2 = n[1]- Z.polupdate(s[0])
