"""
Parameters common to all scripts.

Choices for rho, r, gamma, nu, mubar and sigma and bounds on asset and income
grids are from section F of the appendix to Achdou et al:

    https://benjaminmoll.com/wp-content/uploads/2019/07/HACT_appendix.pdf

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
#Achdou et al uses 2.33 for ind_sd. We used 4 for JEDC paper but that seems
#excessive given that stat distribution of log income is normal.
ind_sd = 3
bnd = [[0, 50], [-ind_sd*nu, ind_sd*nu]]
bnd_NS = [bnd[0], bnd[1], [0, max_age]]
Nexample = (100,15)

maxiter, maxiter_PFI, tol = 20000, 25, 10**-8
show_method, show_iter, show_final = 0, 0, 0

"""
Grid parameters ("true" values and grids for tables and scatterplot)
"""

N_c = 5000
N_true_asset = 5000
income_set = [5,15,25]
N_true_set = [(N_true_asset, I) for I in income_set]

N_sets = [[(25,I), (50,I), (100,I), (250,I), (500,I)] for I in income_set]
N_grid = np.linspace(np.log(50), np.log(500), 10)
N_sets_scatter = [[(int(np.rint(np.exp(n))),I) for n in N_grid] for I in income_set]

relax_list = [10, 50, 100, 200]
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

cols_true = ['$||\Delta c||_1$','$||\Delta c||_{\infty}$',
'$||\Delta c (\%)||_1$','$||\Delta c (\%)||_{\infty}$']

cols_compare = ['$||\Delta c||_1$','$||\Delta c||_{\infty}$',
'$||\Delta c (\%)||_1$','$||\Delta c (\%)||_{\infty}$']

relax_list = relax_list
cols_time = ['VFI']
for relax in relax_list:
    cols_time.append(r'MPFI({0})'.format(relax))
cols_time.append('PFI')

cols_time_nonstat = ['EGM','Seq. PFI','Naive PFI']
cols_time_nonstat_decomp = ['DT policy','DT value','CT policy','CT value']
