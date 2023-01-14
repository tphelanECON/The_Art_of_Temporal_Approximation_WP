"""
Create figures for the example policy and value functions in the stationary
setting (discrete and continuous time).

Systematic exploration of run times and accuracy occurs elsewhere.

Relevant class constructors in classes.py:
    DT_IFP: discrete-time IFP (stationary and age-dependent)
    CT_stat_IFP: continuous-time stationary IFP

The example plots will have the same grid paramters, and therefore do not need
to be indexed by grid or timestep.
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
mubar, sigma, nu = parameters.mubar, parameters.sigma, parameters.nu
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS
show_method, show_iter, show_final = parameters.show_method, parameters.show_iter, parameters.show_final
show_iter=1

N, NA = (500,10), 60
DT_dt = 10**0
CT_dt = 10**-6

X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,N=N,bnd=bnd,NA=NA,
maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
show_final=show_final,dt=DT_dt)
Y = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,N=N,bnd=bnd,
maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
show_final=show_final,dt=CT_dt)

"""
Define dictionaries for discrete-time and continuous-time problems.
"""

DT, CT = {}, {}
pol_method_list = ['EGM']
val_method_list = ['VFI','PFI']
print("Running discrete-time problems")
for pol_method in pol_method_list:
    DT[(pol_method,'VFI')] = X.solve_MPFI(pol_method,0,X.V0)
    DT[(pol_method,'PFI')] = X.solve_PFI(pol_method)
print("Running continuous-time problems")
CT = Y.solve_PFI()

"""
Figures

V,c = DT[(pol_method,val_method)][0:2]
b_prime = (1 + X.dt*X.r)*(X.xx[0] + X.dt*(X.ybar*np.exp(X.xx[1]) - c))

timeit(X.weights_indices(b_prime))

timeit(X.polupdate_EGM(V))
timeit(X.P(c))
timeit(Y.P(c))
timeit(Y.polupdate(V))
rounded, ind2 = X.round_grid(b_prime)
grid = X.grid[0]

timeit(np.array([np.floor(b/X.Delta[0])*X.Delta[0] for b in b_prime]))
timeit(X.round_grid(b_prime)[0])



"""


#[X.grid[0][0]+(np.floor(b/X.Delta[0])-1)*X.Delta[0] for b in b_prime]

#timeit(X.V(c))

"""
Value and policy functions (discrete-time)
"""

for pol_method in pol_method_list:
    for val_method in val_method_list:
        V,c = DT[(pol_method,val_method)][0:2]
        fig, ax = plt.subplots()
        for j in range(N[1]-1):
            color = colorFader(c1,c2,j/(N[1]-1))
            if j in [0,N[1]-2]:
                inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
                ax.plot(X.grid[0], V[:,j], color=color, label="Income {0}".format(inc), linewidth=1)
            else:
                ax.plot(X.grid[0], V[:,j], color=color, linewidth=1)
        plt.xlabel('Assets $b$')
        plt.legend()
        plt.title('Value functions (discrete-time, {0}, {1})'.format(pol_method, val_method))
        destin = '../../main/figures/DT_V_{0}_{1}.eps'.format(pol_method, val_method)
        plt.savefig(destin, format='eps', dpi=1000)
        #plt.show()
        fig, ax = plt.subplots()
        for j in range(N[1]-1):
            color = colorFader(c1,c2,j/(N[1]-1))
            if j in [0,N[1]-2]:
                inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
                ax.plot(X.grid[0], c[:,j], color=color, label="Income {0}".format(inc), linewidth=1)
            else:
                ax.plot(X.grid[0], c[:,j], color=color, linewidth=1)
        plt.xlabel('Assets $b$')
        plt.legend()
        plt.title('Policy functions (discrete-time, {0}, {1})'.format(pol_method, val_method))
        destin = '../../main/figures/DT_c_{0}_{1}.eps'.format(pol_method, val_method)
        plt.savefig(destin, format='eps', dpi=1000)
        ##plt.show()
        plt.close()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], CT[0][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], CT[0][:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Value functions (continuous-time)')
destin = '../../main/figures/CT_V.eps'
plt.savefig(destin, format='eps', dpi=1000)
##plt.show()
plt.close()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], CT[1][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], CT[1][:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Policy functions (continuous-time)')
destin = '../../main/figures/CT_c.eps'
plt.savefig(destin, format='eps', dpi=1000)
##plt.show()
plt.close()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    c, y = CT[1], np.exp(X.grid[1][:])
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], (1+X.r)*(X.grid[0] + y[j] - c[:,j]) - X.grid[0], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], (1+X.r)*(X.grid[0] + y[j] - c[:,j]) - X.grid[0], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Drift (continuous-time)')
destin = '../../main/figures/CT_change_assets.eps'
plt.savefig(destin, format='eps', dpi=1000)
##plt.show()
plt.close()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    c, y = CT[1], np.exp(X.grid[1][:])
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], X.r*X.grid[0] + y[j] - c[:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], X.r*X.grid[0] + y[j] - c[:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Drift (continuous-time)')
destin = '../../main/figures/CT_drift_optimal.eps'
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()
#plt.close()

"""
Now plot the largest that Delta_t can be when evaluated at the optimum? No,
"""

"""
Comparison between discrete and continuous-time
"""

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], DT[('EGM','PFI')][0][:,j] - CT[0][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], DT[('EGM','PFI')][0][:,j] - CT[0][:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Value function differences (discrete minus continuous-time)')
destin = '../../main/figures/DT_CT_compare_V.eps'
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], DT[('EGM','PFI')][1][:,j] - CT[1][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], DT[('EGM','PFI')][1][:,j] - CT[1][:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Policy function differences (discrete minus continuous-time)')
destin = '../../main/figures/DT_CT_compare_c.eps'
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

"""
Now the difference in assets in the stationary environment
"""

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    c, y = DT[(pol_method,'VFI')][1][:,j], np.exp(X.grid[1][:])
    d_assets = (1 + X.dt*X.r)*(X.grid[0] + y[j] - c) - X.grid[0]
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], d_assets, color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], d_assets, color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Difference in assets $\Delta b$ in stationary problem')
destin = '../../main/figures/DT_stat_drift.eps'
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

"""
Now plot the minimum Delta_t for which convergence is assured.


fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    #c, y = DT[(pol_method,'VFI')][1][:,j], np.exp(X.grid[1][:])
    if j in [0,N[1]-2]:
        inc = Y.ybar*np.round(np.exp(Y.grid[1][j]),2)
        ax.plot(Y.grid[0], Y.c0[:,j], color=color, linestyle='dotted', label="Income {0}".format(inc), linewidth=1)
        ax.plot(Y.grid[0], CT[1][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(Y.grid[0], Y.c0[:,j], color=color, linestyle='dotted', label="Income {0}".format(inc), linewidth=1)
        ax.plot(Y.grid[0], CT[1][:,j], color=color, linewidth=1)
plt.xlabel('Assets $b$')
#plt.legend()
#plt.title('Difference in assets $\Delta b$ in stationary problem')
#destin = '../../main/figures/DT_stat_drift.eps'
#plt.savefig(destin, format='eps', dpi=1000)
#plt.show()
"""
