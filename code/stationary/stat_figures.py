"""
Figures for example policy and value functions in the stationary setting.

Systematic exploration of run times and accuracy occurs elsewhere.

This script solves both using PFI. Note that this HAPPENS to converge for
this discrete-time problem but is not ASSURED to do so.

Relevant class constructors in classes.py:
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
show_method, show_iter, show_final = 1,1,1
ybar = parameters.ybar

N = parameters.Nexample
DT_dt = parameters.DT_dt
CT_dt = parameters.CT_dt_true
N_t = parameters.N_t

X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,ybar=ybar,mubar=mubar,sigma=sigma,
N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,show_method=show_method,
show_iter=show_iter,show_final=show_final,dt=DT_dt)

Y = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,ybar=ybar,mubar=mubar,sigma=sigma,
N=N,bnd=bnd,maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
show_final=show_final,dt=CT_dt)

"""
Define dictionaries for discrete-time and continuous-time problems.
"""

prob = 'KD'
DT, CT = {}, {}
#pol_method_list = ['BF','EGM']
pol_method_list = ['EGM']
val_method_list = ['PFI']
print("Running continuous-time problems")
CT = Y.solve_PFI()

print("Running discrete-time problems")
for pol_method in pol_method_list:
    #DT[(pol_method,'MPFI')] = X.solve_MPFI(pol_method,10,X.V0,prob=prob)
    DT[(pol_method,'PFI')] = X.solve_PFI(pol_method,prob=prob)

"""
Figures
"""

"""
Value and policy functions (discrete-time)
"""

for pol_method in pol_method_list:
    for val_method in val_method_list:
        V,c = DT[(pol_method,val_method)][0:2]
        fig, ax = plt.subplots()
        for j in range(N[1]+1):
            color = colorFader(c1,c2,j/(N[1]+1))
            if j in [0,N[1]]:
                inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
                ax.plot(X.grid[0], V[:,j], color=color, label="Income {0}".format(inc), linewidth=1)
            else:
                ax.plot(X.grid[0], V[:,j], color=color, linewidth=1)
        plt.xlabel('Assets $b$')
        plt.legend()
        plt.title('Value functions (discrete-time, {0}, {1})'.format(pol_method, val_method))
        destin = '../../main/figures/DT_V_{0}_{1}.eps'.format(pol_method, val_method)
        plt.savefig(destin, format='eps', dpi=1000)
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        for j in range(N[1]+1):
            color = colorFader(c1,c2,j/(N[1]+1))
            if j in [0,N[1]]:
                inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
                ax.plot(X.grid[0], c[:,j], color=color, label="Income {0}".format(inc), linewidth=1)
            else:
                ax.plot(X.grid[0], c[:,j], color=color, linewidth=1)
        plt.xlabel('Assets $b$')
        plt.legend()
        plt.title('Policy functions (discrete-time, {0}, {1})'.format(pol_method, val_method))
        destin = '../../main/figures/DT_c_{0}_{1}.eps'.format(pol_method, val_method)
        plt.savefig(destin, format='eps', dpi=1000)
        plt.show()
        plt.close()

fig, ax = plt.subplots()
for j in range(N[1]+1):
    color = colorFader(c1,c2,j/(N[1]+1))
    if j in [0,N[1]]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], CT[0][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], CT[0][:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Value functions (continuous-time)')
destin = '../../main/figures/CT_V.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
plt.close()

fig, ax = plt.subplots()
for j in range(N[1]+1):
    color = colorFader(c1,c2,j/(N[1]+1))
    if j in [0,N[1]]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], CT[1][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], CT[1][:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Policy functions (continuous-time)')
destin = '../../main/figures/CT_c.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
plt.close()

fig, ax = plt.subplots()
for j in range(N[1]+1):
    color = colorFader(c1,c2,j/(N[1]+1))
    c, y = CT[1], X.ybar*np.exp(X.grid[1][:])
    if j in [0,N[1]]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], X.r*X.grid[0] + y[j] - c[:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], X.r*X.grid[0] + y[j] - c[:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Drift (continuous-time)')
destin = '../../main/figures/CT_drift_optimal.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
plt.close()

fig, ax = plt.subplots()
V,c = DT[(pol_method,val_method)][0:2]
for j in range(N[1]+1):
    color = colorFader(c1,c2,j/(N[1]+1))
    y = np.exp(X.grid[1][:])
    d_assets = (1+X.dt*X.r)*(X.grid[0] + y[j] - c[:,j]) - X.grid[0]
    if j in [0,N[1]]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], d_assets, color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], d_assets, color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Difference in assets $\Delta b$ in stationary problem')
destin = '../../main/figures/DT_stat_drift.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

"""
Comparison between discrete and continuous-time
"""

fig, ax = plt.subplots()
for j in range(N[1]+1):
    color = colorFader(c1,c2,j/(N[1]+1))
    if j in [0,N[1]]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], DT[('EGM','PFI')][1][:,j] - CT[1][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], DT[('EGM','PFI')][1][:,j] - CT[1][:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Policy function differences')
destin = '../../main/figures/DT_CT_compare_c.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
plt.close()

"""
Now test the timestep
"""

Z = {}
CT_test = {}
#dt_tests = [0.29,0.3,0.31,0.32,0.33]
dt_tests = [0.05, 0.075, 0.09]
N_test = parameters.N_sets[1][-1]
for i in range(len(dt_tests)):
    Z[i] = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,ybar=ybar,mubar=mubar,sigma=sigma,
    N=N_test,bnd=bnd,maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
    show_final=show_final,dt=dt_tests[i])
    print("Running continuous-time problem for dt = {0}".format(dt_tests[i]))
    CT_test[i] = Z[i].solve_PFI()

for i in range(len(dt_tests)):
    print(np.mean(np.abs(CT_test[i][1]-CT_test[0][1])))
