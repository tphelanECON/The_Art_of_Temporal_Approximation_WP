"""
Parameters common to all scripts. Column names for tables also appear here.

The choices for rho, r, gamma, nu, mubar and sigma and the bounds on the asset
and income gridscome are from section F of the appendix to Achdou et al:

    https://benjaminmoll.com/wp-content/uploads/2019/07/HACT_appendix.pdf

I think we want cols_true and cols_compare to be in percentage terms.

cols_true = ['$||c - c_{\textnormal{true}}||_1$',
'$||c - c_{\textnormal{true}}||_{\infty}$',
'$||c/c_{\textnormal{true}}-1||_1$',
'$||c/c_{\textnormal{true}}-1||_{\infty}$']

cols_compare = ['$||c_{DT} - c_{CT}||_1$',
'$||c_{DT} - c_{CT}||_{\infty}$',
'$||c_{DT}/c_{CT} - 1||_1$',
'$||c_{DT}/c_{CT} - 1||_{\infty}$']


"""

import numpy as np
import matplotlib as mpl

"""
Preferences and income parameters
"""
rho, r, gamma, nu = 1/0.95-1, 0.03, 2.0, 0.2
ybar, mubar, sigma = 1, -np.log(0.95), np.sqrt(-2*np.log(0.95))*nu

max_age = 60
NA = 60
NA_scale = int(10)
NA_true = int(NA_scale*NA)
ind_sd = 4
bnd = [[0, 60], [-ind_sd*nu, ind_sd*nu]]
bnd_NS = [bnd[0], bnd[1], [0, max_age]]
Nexample = (100,10)

maxiter, maxiter_PFI, tol = 20000, 25, 10**-8
show_method, show_iter, show_final = 0, 0, 0

"""
Grid parameters ("true" values and grids for tables and scatterplot)
"""

N_true = (10000, 10)
N_c = 5000
N_set = [(50,10), (100,10), (250,10), (500,10), (1000,10)]
N_grid = np.linspace(np.log(50), np.log(1000), 10)
N_set_scatter = [(int(np.rint(np.exp(n))),10) for n in N_grid]

relax_list = [10, 50, 100]
DT_dt = 10**0
CT_dt_true = 10**-6
CT_dt_mid = 10**-3
CT_dt_big = 2*10**-2
N_t = int(1/CT_dt_true)

"""
Columns, plotting and rounding numbers
"""

n_round_acc = 4
n_round_time = 2

c1,c2='lightsteelblue','darkblue'
def colorFader(c1,c2,mix):
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(c1)) + mix*np.array(mpl.colors.to_rgb(c2)))

#I think we want this to be \Delta c and then \Delta c (\%)

cols_true = ['$||\Delta c||_1$','$||\Delta c||_{\infty}$',
'$||\Delta c (\%)||_1$','$||\Delta c (\%)||_{\infty}$']

cols_compare = ['$||\Delta c||_1$','$||\Delta c||_{\infty}$',
'$||\Delta c (\%)||_1$','$||\Delta c (\%)||_{\infty}$']

relax_list = relax_list
cols_time = ['VFI']
for relax in relax_list:
    cols_time.append(r'MPFI ({0})'.format(relax))
cols_time.append('PFI')

cols_time_nonstat = ['EGM','Seq. PFI','Naive PFI']
cols_time_nonstat_decomp = ['DT policy','DT value','CT policy','CT value']
