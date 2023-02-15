"""
Some timing exercises

func1_time = %timeit -o -n 1000 custom_interp()
time_decomp['DT']['Interpolation'] = np.mean(func1_time.timings)
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import classes, parameters
from scipy.interpolate import interp1d

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

N, NA = (400,25), 60
DT_dt = parameters.DT_dt
CT_dt = parameters.CT_dt_true

X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,N=N,bnd=bnd,
NA=NA,maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
show_final=show_final,dt=DT_dt)

Y = classes.CT_stat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,N=N,
bnd=bnd,maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
show_final=show_final,dt=CT_dt)

Z = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,bnd=bnd_NS,
N=(N[0],N[1],NA),maxiter=maxiter,tol=tol,show_method=show_method,
show_iter=show_iter,show_final=show_final)

#V, c, t = X.nonstat_solve('EGM',prob='KD',matrix=False)
#V_, c_, t_ = X.nonstat_solve('EGM',prob='KD',matrix=True)

DT, CT = {}, {}
pol_method_list = ['EGM']
val_method_list = ['VFI','PFI']
model_list = ['Tauchen']

print("Running discrete-time problems")
for pol_method in pol_method_list:
    for prob in model_list:
        DT[(pol_method,'PFI',prob)] = X.solve_PFI(pol_method,prob=prob)

runs = 2
for prob in model_list:
    c_opt = DT[(pol_method,'PFI',prob)][1]
    V0 = X.V(c_opt,prob=prob)
    Vp = X.Vp_cont(V0,prob=prob)
    b_prime = (1 + X.dt*X.r)*(X.xx[0] + X.dt*(X.ybar*np.exp(X.xx[1]) - c_opt))
    P_DT = X.P(c_opt,prob=prob)
    P_CT = Y.P(c_opt)
    bb, zz = X.grid[0][X.ii], X.grid[1][X.jj]
    #candidate asset values and consumption
    cand_b = X.dt*X.u_prime_inv((1+X.dt*X.r)*np.exp(-X.rho*X.dt)*Vp) + bb/(1+X.dt*X.r) - X.dt*X.ybar*zz
    cand_c = (cand_b - bb/(1+X.dt*X.r))/X.dt + X.ybar*np.exp(zz)
    cnew = np.zeros((X.N[0]-1,X.N[1]-1))
    def custom_interp():
        for j in range(X.N[1]-1):
            cnew[:,j] = interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(X.grid[0])
        return cnew

    def custom_interp2():
        [interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(X.grid[0]) for j in range(X.N[1]-1)]
        return cnew

    def solve_lin(P):
        b = c_opt**(1-X.gamma)/(1-X.gamma)
        D = np.exp(-X.rho*X.Dt).reshape((X.M,))
        B = (sp.eye(X.M) - sp.diags(D)*P)/X.dt
        return sp.linalg.spsolve(B, b.reshape((X.M,))).reshape((X.N[0]-1,X.N[1]-1))

    time_decomp = {}
    time_decomp['DT'] = {}
    time_decomp['CT'] = {}
    time_decomp['DT/CT'] = {}

    V0 = X.V(c=c_opt,prob=prob)
    for framework in ['DT','CT']:
        for col in ['Total','Iterations']:
            time_decomp[framework][col] = 0
    for run in range(runs):
        sol = X.solve_PFI(method='EGM',prob=prob)
        time_decomp['DT']['Total'] = sol[2]+time_decomp['DT']['Total']
        time_decomp['DT']['Iterations'] = int(sol[3])+time_decomp['DT']['Iterations']
        sol = Y.solve_PFI()
        time_decomp['CT']['Total'] = sol[2]+time_decomp['CT']['Total']
        time_decomp['CT']['Iterations'] = int(sol[3])+time_decomp['CT']['Iterations']
    for framework in ['DT','CT']:
        for col in ['Total','Iterations']:
            time_decomp[framework][col] = time_decomp[framework][col]/runs
    func1_time = %timeit -o -n 1000 X.polupdate(method='EGM',V=V0,prob=prob,stat=1)
    time_decomp['DT']['Update policy'] = np.mean(func1_time.timings)
    func1_time = %timeit -o -n 1000 Y.polupdate(V=V0)
    time_decomp['CT']['Update policy'] = np.mean(func1_time.timings)
    func1_time = %timeit -o -n 1000 X.Vp_cont(V0,prob)
    time_decomp['DT']['Vp_cont'] = np.mean(func1_time.timings)
    func1_time = %timeit -o -n 1000 X.Vp_cont_jit(V0,prob)
    time_decomp['DT']['Vp_cont_jit'] = np.mean(func1_time.timings)
    #func1_time = %timeit -o -n 1000 X.P(c=c_opt,prob=prob)
    #time_decomp['DT']['Build P'] = np.mean(func1_time.timings)
    #func1_time = %timeit -o -n 1000 Y.P(c=c_opt)
    #time_decomp['CT']['Build P'] = np.mean(func1_time.timings)
    func1_time = %timeit -o -n 1000 solve_lin(P_DT)
    time_decomp['DT']['solve lin. sys.'] = np.mean(func1_time.timings)
    func1_time = %timeit -o -n 1000 solve_lin(P_CT)
    time_decomp['CT']['solve lin. sys.'] = np.mean(func1_time.timings)
    #for framework in ['DT','CT']:
    #    time_decomp[framework]['One iteration'] = time_decomp[framework]['Update policy'] \
    #    + time_decomp[framework]['Build P'] + time_decomp[framework]['solve lin. sys.']
    func1_time = %timeit -o -n 1000 X.MPFI(c_opt,V0,0,prob=prob)
    time_decomp['DT']['Jacobi relax'] = np.mean(func1_time.timings)
    func1_time = %timeit -o -n 1000 X.Jacobi_no_matrix(c_opt,V0,0,prob=prob)
    time_decomp['DT']['Jacobi relax (no matrix)'] = np.mean(func1_time.timings)
    func1_time = %timeit -o -n 1000 Y.MPFI(c_opt,V0,0)
    time_decomp['CT']['Jacobi relax'] = np.mean(func1_time.timings)
    list_1 = list(time_decomp['DT'].keys())
    list_2 = list(time_decomp['CT'].keys())
    list_3 = [l for l in list_1 if l in list_2]
    list_3.remove('Iterations')
    for l in list_3:
        time_decomp['DT/CT'][l] = time_decomp['DT'][l]/time_decomp['CT'][l]
    df = pd.DataFrame(data=time_decomp)
    df = df.round(decimals=4)

    destin = '../../main/figures/time_decomp_{0}.tex'.format(prob)
    with open(destin,'w') as tf:
        tf.write(df.to_latex(escape=False,column_format='lccccc'))


def f1(b):
    PV = np.zeros((X.N[0]-1,X.N[1]-1))
    for i in range(10):
        PV = PV + b
    return PV

def f2(b):
    return sum([b for i in range(10)])

b = (1 + X.dt*X.r)*(X.xx[0] + X.dt*(X.ybar*np.exp(X.xx[1])))

#N = X.N

p = X.p_z[prob]
N = X.N
ii, jj = np.meshgrid(range(N[0]-1),range(N[1]-1),indexing='ij')

#nopython=True
@jit()
def g1(V):
    return sum([p[ii,jj,k]*V[ii,k] for k in range(N[1]-1)])

def g2(V):
    return sum([p[ii,jj,k]*V[ii,k] for k in range(N[1]-1)])

#V_cont = np.zeros((N[0]-1,N[1]-1))
#for k in range(N[1]-1):
#    V_cont = V_cont + p[ii,jj,k]*V[ii,k]

start = time.time()
g1(V0)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
g1(2*V0)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

@jit(nopython=True)
def g2():
    return X.Vp_cont(V0,prob)

timeit(X.Vp_cont(V0,prob))

start = time.time()
g2()
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

start = time.time()
g2()
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

def g2():
    return sum([X.p_z[prob][X.ii,X.jj,k]*V0[X.ii,k] for k in range(X.N[1]-1)])

timeit(g1())
timeit(g2())



alpha, ind = X.weights_indices(b_prime)
ii, jj = X.ii, X.jj
if prob=='KD':
    for key in [-1,0,1]:
        #lower then upper weights on asset transition:
        row, col = ii*(X.N[1]-1) + jj, ind[ii,jj]*(X.N[1]-1) + jj + key
        ind_bnd = (col>-1)*(col<X.M)*(jj + key > -1)*(jj + key < X.N[1]-1)
        ii_, jj_ = ii[ind_bnd], jj[ind_bnd]
        ind_ = ind[ii_,jj_]
        alpha_, prob_ = alpha[ii_,jj_], X.probs[prob][key][ii_,jj_]
        PV[ii_,jj_] = PV[ii_,jj_] + alpha_*prob_*V[ind_, jj_ + key] + (1-alpha_)*prob_*V[ind_+1, jj_ + key]

prob = 'KD'

PV = np.zeros((X.N[0]-1,X.N[1]-1))
V, c = V0, c_opt
b_prime = (1 + X.dt*X.r)*(X.xx[0] + X.dt*(X.ybar*np.exp(X.xx[1]) - c))
alpha, ind = X.weights_indices(b_prime)
ii, jj = X.ii, X.jj
if prob=='KD':
    for key in [-1,0,1]:
        #lower then upper weights on asset transition:
        row, col = ii*(X.N[1]-1) + jj, ind[ii,jj]*(X.N[1]-1) + jj + key
        ind_bnd = (col>-1)*(col<X.M)*(jj + key > -1)*(jj + key < X.N[1]-1)
        row_, col_ = row[ind_bnd], col[ind_bnd]
        ii_, jj_ = ii[ind_bnd], jj[ind_bnd]
        ind_ = ind[ii_,jj_]
        alpha_, prob_ = alpha[ii_,jj_], X.probs[prob][key][ii_,jj_]
        #following is contribution to PV
        #PV_dict[key][ii_, jj_] = alpha_*prob_*V[ind_, jj_ + key] + (1-alpha_)*prob_*V[ind_+1, jj_ + key]
        PV[ii_,jj_] = PV[ii_,jj_] + alpha_*prob_*V[ind_, jj_ + key] + (1-alpha_)*prob_*V[ind_+1, jj_ + key]



V_1 = X.MPFI(c_opt,V0,0,prob=prob)
V_2 = X.dt*X.u(c) + np.exp(-X.rho*X.dt)*PV

diff = np.max(np.abs(V_1-V_2))

#V_2 = X.Jacobi_no_matrix(c_opt,V0,0,prob=prob)

'KD'
timeit(X.MPFI(c_opt,V0,0,prob=prob))
timeit(X.Jacobi_no_matrix(c_opt,V0,0,prob=prob))

runs = 5
time_decomp = {}
time_decomp['DT'] = {}
time_decomp['CT'] = {}
time_decomp['DT/CT'] = {}

for framework in ['DT','CT']:
    for col in ['Total','Iterations']:
        time_decomp[framework][col] = []
for run in range(runs):
    print(run)
    sol = X.solve_PFI(method='EGM',prob=prob)
    time_decomp['DT']['Total'].append(sol[2])
    time_decomp['DT']['Iterations'].append(int(sol[3]))
    sol = Y.solve_PFI()
    time_decomp['CT']['Total'].append(sol[2])
    time_decomp['CT']['Iterations'].append(int(sol[3]))

means, stds = {}, {}
for framework in ['DT','CT']:
    means[framework] = np.mean(np.array(time_decomp[framework]['Total']))
    stds[framework] = np.std(np.array(time_decomp[framework]['Total']))

#for col in ['Total','Iterations']:
#    time_decomp[framework][col] = time_decomp[framework][col]/runs
