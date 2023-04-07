"""
Create figures for non-stationary problem without medical expenditures.

Only quantity depicted here is change in assets in initial period.
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import matplotlib.pyplot as plt
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
ybar = parameters.ybar

show_iter, show_method, show_final = 1, 1, 1
NA = parameters.NA
N = parameters.Nexample
DT_dt = parameters.DT_dt
CT_dt = parameters.CT_dt_true
N_t = parameters.N_t

X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,ybar=ybar,mubar=mubar,sigma=sigma,
N=N,NA=NA,N_t=N_t,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt)

Z = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,ybar=ybar,mubar=mubar,sigma=sigma,
bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,tol=tol,show_method=show_method,
show_iter=show_iter,show_final=show_final)

DT, CT, mean_time = {}, {}, {}
DT['VFI_EGM'] = X.nonstat_solve('EGM')
V_DT, c_DT = DT['VFI_EGM'][0:2]
CT['seq_PFI'] = Z.solve_seq_imp()
V_imp, c_imp = CT['seq_PFI'][0:2]

"""
Following is to remind myself that it can make a big difference in run times
if we avoid constructing a sparse matrix when solving the problem:

    DT['matrix'] = X.nonstat_solve('EGM',matrix=True)
    DT['no_matrix'] = X.nonstat_solve('EGM',matrix=False)

"""

dc_p = 100*np.abs((c_imp-c_DT)/c_DT)

print("Consumption EGM + VFI vs CT + implicit MCA in initial period")
print("Max percent difference:", np.max(dc_p[:,:,0]))
print("Mean percent difference:", np.mean(dc_p[:,:,0]))

"""
Change in assets in initial period
"""

fig, ax = plt.subplots()
for j in range(N[1]+1):
    color = colorFader(c1,c2,j/(N[1]+1))
    c, y = c_DT[:,j,0], np.exp(X.grid[1][:])
    d_assets = (1+X.dt*X.r)*(X.grid[0] + y[j] - c) - X.grid[0]
    if j in [0,N[1]]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], d_assets, color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], d_assets, color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Difference in assets $\Delta b$ in initial period')
destin = '../../main/figures/DT_NS_initial_drift.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
