"""
classes.py shows how to solve the dynamic programming problem when the log
income transition matrix emerges from N_t iterates of the Kushner and Dupuis
transition probabilities. This script checks that the computed policy functions
are not overly sensitive to changes in N_t by comparing with continuous-time case. 

Recall relevant class constructors in classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent)
    CT_stat_IFP: continuous-time stationary IFP

"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import matplotlib.pyplot as plt
import classes, parameters

"""
Get parameters from parameters.py.
"""

c1,c2 = parameters.c1,parameters.c2
colorFader = parameters.colorFader

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS
show_method, show_iter, show_final = parameters.show_method, parameters.show_iter, parameters.show_final
show_final = 1

N, NA = (200,20), 60
DT_dt = parameters.DT_dt
CT_dt = parameters.CT_dt_true

N_t_list = [1,2,5,10]
X, Y, Z = {}, {}, {}
sol = {}
sol['DT'], sol['CT'] = {}, {}
for N_t in N_t_list:
    X[N_t] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,N=N,bnd=bnd,
    NA=NA,N_t=N_t,maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
    show_final=show_final,dt=DT_dt)
    Y[N_t] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,N=N,
    bnd=bnd,maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
    show_final=show_final,dt=CT_dt)
    sol['DT'][N_t] = X[N_t].solve_PFI('EGM',prob='KD')
    sol['CT'][N_t] = Y[N_t].solve_PFI()


for N_t in N_t_list:
    fig, ax = plt.subplots()
    for j in range(X[N_t].N[1]-1):
        color = colorFader(c1,c2,j/(N[1]-1))
        if j in [0,X[N_t].N[1]-2]:
            inc = X[N_t].ybar*np.round(np.exp(X[N_t].grid[1][j]),2)
            ax.plot(X[N_t].grid[0], sol['DT'][N_t][1][:,j] - sol['CT'][N_t][1][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(X[N_t].grid[0], sol['DT'][N_t][1][:,j] - sol['CT'][N_t][1][:,j], color=color, linewidth=1)
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('DT versus CT $N_t$ = {0}'.format(N_t))
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    for j in range(X[N_t].N[1]-1):
        color = colorFader(c1,c2,j/(N[1]-1))
        if j in [0,X[N_t].N[1]-2]:
            inc = X[N_t].ybar*np.round(np.exp(X[N_t].grid[1][j]),2)
            ax.plot(X[N_t].grid[0], sol['DT'][N_t][1][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(X[N_t].grid[0], sol['DT'][N_t][1][:,j], color=color, linewidth=1)
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Policy functions $N_t$ = {0}'.format(N_t))
    plt.show()
    plt.close()
