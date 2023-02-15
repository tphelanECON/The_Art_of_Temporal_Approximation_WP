"""
Create figures for nonstationary problems (discrete and continuous time)
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import time, scipy, scipy.optimize
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy.interpolate import interp1d
import classes, parameters

c1,c2 = parameters.c1,parameters.c2
colorFader = parameters.colorFader

"""
Set parameters
"""

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

N_true, N_c = parameters.N_true, parameters.N_c
show_iter, show_method, show_final = 1, 1, 1
N_true, N_c = parameters.N_true, parameters.N_c
n_round_acc = parameters.n_round_acc
n_round_time = parameters.n_round_time
CT_dt_true = parameters.CT_dt_true
CT_dt_mid = parameters.CT_dt_mid
CT_dt_big = parameters.CT_dt_big
DT_dt = parameters.DT_dt
NA = parameters.NA

N = (100,20)
X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
show_method=show_method,show_iter=show_iter,show_final=show_final,dt=1,NA=NA)
Z = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,bnd=bnd_NS,
N=(N[0],N[1],NA),maxiter=maxiter,tol=tol,show_method=show_method,
show_iter=show_iter,show_final=show_final)

DT, CT = {}, {}
time, time_array = {}, {}
time_ave, time_array_ave = {}, {}
time_decomp = {}
for framework in ['DT','CT']:
    time[framework] = 0
    time_array[framework] = np.zeros((2,NA-1))

runs = 20
for i in range(runs):
    DT[('Tauchen',i)] = X.nonstat_solve('EGM','Tauchen')
    time['DT'] += DT[('Tauchen',i)][2]
    time_array['DT'] += DT[('Tauchen',i)][3]
    CT[('seq_PFI',i)] = Z.solve_seq_imp()
    time['CT'] += CT[('seq_PFI',i)][2]
    time_array['CT'] += CT[('seq_PFI',i)][3]

for framework in ['DT','CT']:
    time_ave[framework] = time[framework]/runs
    time_array_ave[framework] = time_array[framework]/runs
    time_decomp[framework] = time_array_ave[framework].sum(axis=1)
    print("Total time taken for {0}:".format(framework), time_ave[framework])
    print("Sum of components:", time_decomp[framework][0]+ time_decomp[framework][1])
    print("Updating policy:".format(framework), time_decomp[framework][0])
    print("Updating value:".format(framework), time_decomp[framework][1])
