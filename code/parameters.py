"""
Parameters common to all scripts.

Perhaps put column lists here? At the moment, no. I think we want a different
n_round for time and accuracy.

Use mubar not mu.
"""

import numpy as np
import matplotlib as mpl

rho, r, gamma, nu = 0.05, 0.02, 2.0, 0.2
mubar, sigma = -np.log(0.95), np.sqrt(-2*np.log(0.95))*nu
maxiter, maxiter_PFI, tol = 20000, 25, 10**-8

max_age = 60
NA = 60
NA_scale = int(10)
NA_true = int(NA_scale*NA)
bnd = [[0, 60], [-4*nu, 4*nu]]
bnd_NS = [bnd[0], bnd[1], [0, max_age]]

show_method, show_iter, show_final = 0, 0, 0
N_true = (3000, 10)
N_set = [(50,10), (100,10), (250,10), (500,10), (1000,10)]
relax_list = [10,50,100]

N_c = 4000
n_round_acc = 4
n_round_time = 2
DT_dt = 10**0
CT_dt_true = 10**-6
CT_dt_mid = 10**-3
CT_dt_big = 2*10**-2

c1,c2='lightsteelblue','darkblue'
def colorFader(c1,c2,mix):
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(c1)) + mix*np.array(mpl.colors.to_rgb(c2)))
