"""
Comparison of naive PFI and sequential PFI.

This is only to provide us confidence in the code and the sparse solver. These
two approaches literally solve the same system of equations and so the
difference between them ought to be minuscule.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import time, classes, parameters

c1,c2 = parameters.c1, parameters.c2
colorFader = parameters.colorFader

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mu, sigma = parameters.mu, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

show_iter, show_method, show_final = 1, 1, 1
N_true, N_c, n_round = parameters.N_true, parameters.N_c, 4
NA = parameters.NA
N_true = (50,10)

cols_compare = ['$||c_{N} - c_{\textnormal{seq}}||_1$',
'$||c_{N} - c_{\textnormal{seq}}||_{\infty}$',
'$||V_{N} - V_{\textnormal{seq}}||_1$',
'$||V_{N} - V_{\textnormal{seq}}||_{\infty}$']

"""
Functions for data construction and table creation
"""

DT_dt=1
def accuracy_data(N_set,CT_dt):
    """
    Pre-allocate all quantities
    """
    Z, naive, seq, naive_seq = {}, {}, {}, {}
    data_compare = []
    for N in N_set:
        print("Number of gridpoints:", N)
        d_compare = {}
        Z[N] = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mu=mu,sigma=sigma,
        bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,maxiter_PFI=maxiter_PFI,
        tol=tol,show_method=show_method,show_iter=show_iter,show_final=show_final,dt=CT_dt)
        naive[N] = Z[N].solve_PFI()
        seq[N] = Z[N].solve_seq_imp()
        def compare(f1,f2):
            f2_big = np.zeros((X['True'].N[0]-1,X['True'].N[1]-1))
            for j in range(X['True'].N[1]-1):
                f2_big[:,j] = interp1d(X[N].grid[0], f2[:,j],fill_value="extrapolate")(X['True'].grid[0])
            return f1-f2_big
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

    destin = '../main/figures/naive_seq_accuracy.tex'
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='cccccc'))

"""
Create tables
"""

CT_dt = 10**-2
N_set = [(50,10), (100,10), (150,10), (200,10)]
accuracy_tables(N_set, CT_dt)
