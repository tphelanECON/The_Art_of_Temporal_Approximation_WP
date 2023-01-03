"""
Run times for nonstationary problem (discrete and continuous time).

1. EGM + VFI for the discrete-time problem;
2. sequential PFI for the continuous-time problem;
3. naive PFI for the continuous-time problem;
4. value function iteration for the continuous-time problem.

For the time being I will omit 4.
"""

import numpy as np
import pandas as pd
import classes, parameters, time

c1,c2 = parameters.c1, parameters.c2
colorFader = parameters.colorFader

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mu, sigma = parameters.mu, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 1, 1, 1
N_true, N_c, n_round = parameters.N_true, parameters.N_c, 3
NA = parameters.NA

cols = ['EGM','Seq. PFI','Naive PFI']

def time_data(N_set,DT_dt,CT_dt,runs):
    df = pd.DataFrame(data=0,index=N_set,columns=cols)
    for i in range(runs):
        time_data = []
        X, Z = {}, {}
        for N in N_set:
            print("Number of gridpoints:", N)
            d = {} #this is where temporary data is stored
            X[N] = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
            N=N,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
            show_method=show_method,show_iter=show_iter,show_final=show_final,dt=DT_dt,NA=NA)
            Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
            bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
            tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
            d[cols[0]] = X[N].nonstat_solve('EGM')[2]
            d[cols[1]] = Z[N].solve_seq_imp()[2]
            if N[0] < 10:
                d[cols[2]] = Z[N].solve_PFI()[2]
            else:
                d[cols[2]] = 0
            time_data.append(d)
        df = df + pd.DataFrame(data=time_data,index=N_set,columns=cols)
    return df.round(decimals=n_round)/runs

def time_tables(N_set,DT_dt,CT_dt,runs):
    df = time_data(N_set,DT_dt,CT_dt,runs)
    df.index.names = ['Grid size']

    destin = '../main/figures/DT_CT_speed_nonstat.tex'
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

"""
Create tables. Recall arguments: (N_set,DT_dt,CT_dt).
"""

N_set = parameters.N_set
DT_dt = 1.0
CT_dt = 10**-2
runs = 1
#time_tables(parameters.N_set,DT_dt,CT_dt,runs)
df = time_data(N_set,DT_dt,CT_dt,runs)
