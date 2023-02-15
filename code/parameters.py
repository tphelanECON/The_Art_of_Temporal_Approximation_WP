"""
Parameters common to all scripts. Column names for tables also appear here.

The choices for rho, r, gamma, nu, mubar and sigma and the bounds on the asset
and income gridscome from section F of the appendic to Achdou et al:

    https://benjaminmoll.com/wp-content/uploads/2019/07/HACT_appendix.pdf

Remaining choices somewhat arbitrary but inessential.
"""

import numpy as np
import matplotlib as mpl

rho, r, gamma, nu = 1/0.95-1, 0.03, 2.0, 0.2
mubar, sigma = -np.log(0.95), np.sqrt(-2*np.log(0.95))*nu
maxiter, maxiter_PFI, tol = 20000, 25, 10**-8

max_age = 60
NA = 60
NA_scale = int(10)
NA_true = int(NA_scale*NA)
N_t = 10
bnd = [[0, 50], [-4*nu, 4*nu]]
bnd_NS = [bnd[0], bnd[1], [0, max_age]]

show_method, show_iter, show_final = 0, 0, 0
N_true = (4000, 10)
N_set = [(50,10), (100,10), (250,10), (500,10), (1000,10)]
relax_list = [10, 50, 100]

N_c = 5000
n_round_acc = 4
n_round_time = 2
DT_dt = 10**0
CT_dt_true = 10**-6
CT_dt_mid = 10**-3
CT_dt_big = 2*10**-2

c1,c2='lightsteelblue','darkblue'
def colorFader(c1,c2,mix):
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(c1)) + mix*np.array(mpl.colors.to_rgb(c2)))

cols_true = ['$||c - c_{\textnormal{true}}||_1$',
'$||c - c_{\textnormal{true}}||_{\infty}$',
'$||V - V_{\textnormal{true}}||_1$',
'$||V - V_{\textnormal{true}}||_{\infty}$']

cols_compare = ['$||c_{DT} - c_{CT}||_1$',
'$||c_{DT} - c_{CT}||_{\infty}$',
'$||V_{DT} - V_{CT}||_1$',
'$||V_{DT} - V_{CT}||_{\infty}$']

relax_list = relax_list
cols_time = ['VFI']
for relax in relax_list:
    cols_time.append(r'MPFI ({0})'.format(relax))
cols_time.append('PFI')

cols_time_nonstat = ['EGM','Seq. PFI','Naive PFI']
cols_time_nonstat_decomp = ['DT policy','DT value','CT policy','CT value']
