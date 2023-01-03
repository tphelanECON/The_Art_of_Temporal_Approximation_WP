"""
Parameters common to all scripts
"""

import numpy as np
import matplotlib as mpl

rho, r, gamma, nu = 0.05, 0.02, 2.0, 0.2
mu, sigma = -np.log(0.95), np.sqrt(-2*np.log(0.95))*nu
maxiter, maxiter_PFI, tol = 10000, 25, 10**-8

max_age = 60
NA = 60
bnd = [[0, 60], [-4*nu, 4*nu]]
bnd_NS = [bnd[0], bnd[1], [0, max_age]]

show_method, show_iter, show_final = 0, 0, 0
N_true = (5000, 10)
N_set = [(50,10), (100,10), (250,10), (500,10), (1000,10)]


N_c = 5000
n_round = 3

c1,c2='lightsteelblue','darkblue'
def colorFader(c1,c2,mix):
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(c1)) + mix*np.array(mpl.colors.to_rgb(c2)))
