"""
Comparison of naive PFI and sequential PFI.

Serves only to provide check on the code.

Naive PFI and sequential PFI literally solve the same system of equations and
so the difference between the values they return ought to be very small.
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import time, classes, parameters

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 1, 1, 1
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
        bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
        show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        naive[N] = Z[N].solve_PFI()
        seq[N] = Z[N].solve_seq_imp()
        d = (naive[N][1]-seq[N][1])[:,:,:-1]
        d_percent = 100*d/(naive[N][1][:,:,:-1])
        #compare naive- and sequential PFI output
        d_compare[cols_compare[0]] = np.mean(np.abs(d))
        d_compare[cols_compare[1]] = np.max(np.abs(d))
        d_compare[cols_compare[2]] = np.mean(np.abs(d_percent))
        d_compare[cols_compare[3]] = np.max(np.abs(d_percent))
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

N_set = [(50,10),(100,10),(150,10),(200,10)]
accuracy_tables(N_set,parameters.CT_dt_true)
#data_compare = accuracy_data(N_set,parameters.CT_dt_true)
