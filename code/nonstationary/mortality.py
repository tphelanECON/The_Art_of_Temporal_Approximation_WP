"""
Class constructors for the mortality choice example from "The Art of Temporal Approximation".

Author: Thomas Phelan.
Email: tom.phelan@clev.frb.org.

This script contains a single class constructor:

    1. CT_nonstat_IFP_mortality: continuous-time nonstationary (age-dependent) IFP

Some problems that can arise for certain parameters:

    * be careful that asset bounds are chosen s.t. agent wants to dissave at
    the upper bound. Otherwise we get a "spike" in consumption.
    * be careful to not have nonzero outer transitions on the boundary.
    * once we include b=0, if agent doesn't work at lower bound self.u will
    generate nans. They will wish to work at this point though.

Mortality rates obtained from https://www.ssa.gov/oact/STATS/table4c6.html
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
from numba import jit
import scipy.sparse as sp
from scipy.stats import norm
from scipy.sparse import linalg
from scipy.sparse import diags
from scipy.interpolate import interp1d
import itertools, time
import scipy, scipy.optimize
import matplotlib.pyplot as plt
import parameters

"""
Continuous-time income fluctuation problem with:
    * medical expenditures and endogenous lifespan
    * discrete labor-leisure choice
    * lifecycle aspect (certain death at upper bound Abar)
"""

class CT_nonstat_IFP_mortality(object):
    def __init__(self, rho=0.05, r=0.03, gamma=2., ybar=1, mubar=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, bnd=[[0,60],[-0.6,0.6],[0,60]], N=(40,10,10),
    dt = 10**(-6), tol=10**-6, maxiter=200, maxiter_PFI=25, mono_tol=10**(-6),
    show_method=1, show_iter=1, show_final=1, kappac=2, D=-40, mbar = 4,
    Lam0low=0.05, Lam0high=0.05, Lam1low=10, Lam1high=10, etalow=0.9, etahigh=0.8):
        self.r, self.rho, self.gamma = r, rho, gamma
        self.etalow, self.etahigh = etalow, etahigh
        self.ybar, self.mubar, self.sigma = ybar, mubar, sigma
        self.tol, self.mono_tol = tol, mono_tol
        self.maxiter, self.maxiter_PFI = maxiter, maxiter_PFI
        self.N, self.M, self.M_slice = N, (N[0]+1)*(N[1]+1)*(N[2]+1), (N[0]+1)*(N[1]+1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.grid = [np.linspace(self.bnd[i][0],self.bnd[i][1],self.N[i]+1) for i in range(3)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        self.ii, self.jj, self.kk = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),range(self.N[2]+1), indexing='ij')
        self.sigsig = self.sigma*(self.jj>0)*(self.jj<self.N[1])
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1)]
        self.trans_keys_slice = [(1,0),(-1,0),(0,1),(0,-1)]
        self.c0 = self.r*self.xx[0] + self.ybar*np.exp(self.xx[1])
        self.dt, self.Dt = dt, dt + 0*self.c0
        self.kappac = kappac
        self.cmax = 1.5*self.xx[0][-1] + 0*self.c0
        self.show_method, self.show_iter, self.show_final = show_method, show_iter, show_final
        #additional parameters relevant for mortality choice
        self.D, self.mbar = D, mbar
        self.Lam0low, self.Lam0high = Lam0low, Lam0high
        self.Lam1low, self.Lam1high = Lam1low, Lam1high
        #Lam0 and Lam1 will now be vectors, to incorporate age-dependent mortality risk.
        #assume unconditional death rate is cubic in age and other medical
        #parameters (eta and Lam1) are linear in age
        self.DIFF = (self.grid[2]-self.grid[2][0])/(self.grid[2][-1]-self.grid[2][0])
        self.Lam0 = self.Lam0low + self.DIFF**3*(self.Lam0high-self.Lam0low)
        self.Lam1 = self.Lam1low + self.DIFF*(self.Lam1high-self.Lam1low)
        self.eta = self.etalow + self.DIFF*(self.etahigh-self.etalow)

    def u(self,c,l,k):
        c = np.maximum(c,10**(-8))
        scale = 1*(l==0) + self.eta[k]*(l==1)
        if self.gamma==1:
            return np.log(c*scale)
        else:
            return (c*scale)**(1-self.gamma)/(1-self.gamma)

    def lam_func(self,m,k,form):
        if form=='nonlinear':
            return self.Lam0[k]*np.exp(-self.Lam1[k]*m)
        else:
            return np.maximum(0,self.Lam0[k]*(1-self.Lam1[k]*m))

    def p_func_slice(self,ind,pol,k,form):
        ii,jj = ind
        p_func_diff = {}
        x = (self.xx[0][ii,jj,0],self.xx[1][ii,jj,0])
        dt, sig = self.Dt[ii,jj,0], self.sigsig[ii,jj,0]
        c, m, l = pol[0][ii,jj], pol[1][ii,jj], pol[2][ii,jj]
        denom = 1 - self.lam_func(m,k,form)*dt
        p_func_diff[(1,0)] = (dt/self.Delta[0])*np.maximum(self.r*x[0]+self.ybar*np.exp(x[1])*l-c,0)/denom
        p_func_diff[(-1,0)] = (dt/self.Delta[0])*(np.maximum(-(self.r*x[0]+self.ybar*np.exp(x[1])*l-c),0)+m)/denom
        p_func_diff[(0,1)] = (dt/self.Delta[1]**2)*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-x[1]),0))/denom
        p_func_diff[(0,-1)] = (dt/self.Delta[1]**2)*(sig**2/2 + self.Delta[1]*np.maximum(-self.mubar*(-x[1]),0))/denom
        p_func_diff[(0,0,1)] = (dt/self.Delta[2])/denom
        #overwrite above definition at upper bound of assets:
        drift = self.r*x[0]+self.ybar*np.exp(x[1])*l-c-m
        up = (dt/self.Delta[0])*np.maximum(drift,0)/denom
        down = (dt/self.Delta[0])*np.maximum(-drift,0)/denom
        p_func_diff[(1,0)][ii==self.N[0]] = up[ii==self.N[0]]
        p_func_diff[(-1,0)][ii==self.N[0]] = down[ii==self.N[0]]
        return p_func_diff

    #transition matrix at age k CONDITIONAL on survival:
    def P_slice(self,pol,k,form):
        ii, jj = np.meshgrid(range(self.N[0]+1), range(self.N[1]+1), indexing='ij')
        row = ii*(self.N[1]+1) + jj
        diag = 1 - sum(self.p_func_slice((ii,jj),pol,k,form).values())
        P = self.P_func_slice(row,row,diag)
        for key in self.trans_keys_slice:
            column = row + key[0]*(self.N[1]+1) + key[1]
            P = P + self.P_func_slice(row,column,self.p_func_slice((ii,jj),pol,k,form)[key])
        if np.min(P.todense()) < 0:
            print("Probabilieis negative!")
        return P

    def V_imp(self,V_imp_up,pol_slice,k,form):
        Dt_slice = self.Dt[:,:,0]
        c_slice, m_slice, l_slice = pol_slice
        b = Dt_slice*self.u(c_slice,l_slice,k) \
        + np.exp(-self.rho*Dt_slice)*Dt_slice*(self.lam_func(m_slice,k,form)*self.D + V_imp_up/self.Delta[2])
        D = (np.exp(-self.rho*Dt_slice)*(1 - self.lam_func(m_slice,k,form)*Dt_slice)).reshape((self.M_slice,))
        B = sp.eye(self.M_slice) - sp.diags(D)*self.P_slice(pol_slice,k,form)
        b, B = b/self.dt, B/self.dt
        return sp.linalg.spsolve(B, b.reshape((self.M_slice,))).reshape((self.N[0]+1,self.N[1]+1))

    #need to be careful about the behavior of the following on boundaries.
    def polupdate_slice(self,V_imp,k,form):
        Vbig = -10**6*np.ones((self.N[0]+3, self.N[1]+3))
        Vbig[1:-1,1:-1] = V_imp
        VB1 = (Vbig[1:-1,1:-1]-Vbig[:-2,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1]-Vbig[1:-1,1:-1])/self.Delta[0]
        with np.errstate(divide='ignore',invalid='ignore'):
            if form=='nonlinear':
                m_star = np.log(self.Lam0[k]*self.Lam1[k]*(V_imp - self.D)/VB1)/self.Lam1[k]
                m_star[(V_imp - self.D)/VB1 < 0] = 0
            else:
                high_m = self.lam_func(self.mbar,k,form)*(self.D-V_imp) - self.mbar*VB1
                low_m = self.lam_func(0,k,form)*(self.D-V_imp) - 0*VB1
                m_star = self.mbar*(high_m>low_m)
        m = np.maximum(0,np.minimum(m_star,self.mbar))
        cmax, Dt_slice = self.cmax[:,:,0], self.Dt[:,:,0]
        drift_no_c = np.zeros((self.N[0]+1, self.N[1]+1, 2))
        C = np.zeros((self.N[0]+1, self.N[1]+1, 2))
        payoff = np.zeros((self.N[0]+1, self.N[1]+1, 2))
        b, y = self.xx[0][:,:,0], self.ybar*np.exp(self.xx[1][:,:,0])
        #compute optimal c for each choice of l and then tae ptwise max:
        for l in [0,1]:
            drift_no_c[:,:,l] = self.r*b + y*l
            #recall that we altered transition probabilities on upper asset bound:
            drift_no_c[-1,:,l] = self.r*b[-1,:] + y[-1,:]*l - m[-1,:]
            obj = lambda c: self.u(c,l,k) + np.exp(-self.rho*Dt_slice) \
            *(np.maximum(drift_no_c[:,:,l] - c, 0)*VF1 - np.maximum(-(drift_no_c[:,:,l] - c), 0)*VB1)
            with np.errstate(divide='ignore',invalid='ignore'):
                #candidate minima on [0, drift_no_c] and [drift_no_c, infty], resp.
                pref_shift = (l==1)*self.eta[k]**(self.gamma-1) + (l==0)*1
                clow = np.minimum((pref_shift*np.exp(-self.rho*Dt_slice)*VF1)**(-1/self.gamma), drift_no_c[:,:,l])
                chigh = np.maximum((pref_shift*np.exp(-self.rho*Dt_slice)*VB1)**(-1/self.gamma), drift_no_c[:,:,l])
            clow[VF1<=0], chigh[VB1<=0] = cmax[VF1<=0], cmax[VB1<=0]
            runmax = np.concatenate((obj(drift_no_c[:,:,l]).reshape(1,self.M_slice), \
            obj(clow).reshape(1,self.M_slice), obj(chigh).reshape(1,self.M_slice)))
            IND = np.argmax(runmax,axis=0).reshape(self.N[0]+1,self.N[1]+1)
            C[:,:,l] = (IND==0)*drift_no_c[:,:,l] + (IND==1)*clow + (IND==2)*chigh
            C[0,:,l] = np.minimum(C[0,:,l], drift_no_c[0,:,l] - m[0,:])
            C[-1,:,l] = np.maximum(C[-1,:,l], drift_no_c[-1,:,l])
            payoff[:,:,l] = obj(C[:,:,l])
        work = payoff[:,:,1]>payoff[:,:,0]
        c = work*C[:,:,1] + (1-work)*C[:,:,0]
        return c, m, work

    def solveVslice(self,V_imp_up,pol_guess,k,form):
        V = self.V_imp(V_imp_up,pol_guess,k,form)
        eps, i = 1, 1
        while i < 20 and eps > self.tol:
            c,m,l = self.polupdate_slice(V,k,form)
            V1 = self.V_imp(V_imp_up,self.polupdate_slice(V,k,form),k,form)
            eps = np.amax(np.abs(V - V1))
            V, i = V1, i+1
        #print(i,eps)
        return V

    def solve_seq_imp(self,form='nonlinear'):
        if self.show_method==1:
            print("Starting sequential PFI")
        V = np.zeros((self.N[0]+1,self.N[1]+1,self.N[2]+1))
        c = np.zeros((self.N[0]+1,self.N[1]+1,self.N[2]+1))
        m = np.zeros((self.N[0]+1,self.N[1]+1,self.N[2]+1))
        l = np.zeros((self.N[0]+1,self.N[1]+1,self.N[2]+1))
        time_array = np.zeros((2,self.N[2]+1))
        pol_guess = (self.c0[:,:,0], 0*self.c0[:,:,0], 1+0*self.c0[:,:,0])
        V_slice_cont = self.D*np.ones((self.N[0]+1,self.N[1]+1))
        V[:,:,self.N[2]-1] = self.solveVslice(V_slice_cont, pol_guess,self.N[2]-1,form)
        c[:,:,self.N[2]-1], m[:,:,self.N[2]-1], l[:,:,self.N[2]-1] = self.polupdate_slice(V[:,:,self.N[2]-1],self.N[2]-1,form)
        tic = time.time()
        for k in range(self.N[2]-1):
            if (int(self.N[2]-1-k-1) % 10 == 0):
                print("Age:", self.N[2]-1-k-1)
                print("Failure of concavity?", "Occurs a fraction", np.mean(self.Vpp_check(V[:,:,self.N[2]-1-k])), "of the time")
            tV1 = time.time()
            pol = (c[:,:,self.N[2]-1-k], m[:,:,self.N[2]-1-k], l[:,:,self.N[2]-1-k])
            V[:,:,self.N[2]-1-k-1] = self.solveVslice(V[:,:,self.N[2]-1-k],pol,self.N[2]-1-k-1,form)
            tV2 = time.time()
            tc1 = time.time()
            new_pol = self.polupdate_slice(V[:,:,self.N[2]-1-k-1],self.N[2]-1-k-1,form)
            c[:,:,self.N[2]-1-k-1] = new_pol[0]
            m[:,:,self.N[2]-1-k-1] = new_pol[1]
            l[:,:,self.N[2]-1-k-1] = new_pol[2]
            tc2 = time.time()
            time_array[:,self.N[2]-1-k-1] = tc2-tc1, tV2-tV1
        toc = time.time()
        if self.show_final==1:
            print("Time taken:", toc-tic)
        drift = self.r*self.xx[0] + self.ybar*np.exp(self.xx[1])*l - c - m
        print("Maximum drift at upper asset bound negative?", np.max(drift[-1,:,:-1])<0)
        return V, (c,m,l), toc-tic, time_array

    def P_func_slice(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M_slice)],B[(B>-1)*(B<self.M_slice)],C[(B>-1)*(B<self.M_slice)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M_slice,self.M_slice))

    def P_func(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M)],B[(B>-1)*(B<self.M)],C[(B>-1)*(B<self.M)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def Vpp_check(self,V_slice):
        Vbig = -10**6*np.ones((self.N[0]+3, self.N[1]+1))
        Vbig[1:-1,:] = V_slice
        Vpp = (Vbig[2:,:] - 2*Vbig[1:-1,:] + Vbig[:-2,:])/self.Delta[0]**2
        return Vpp > 10**(-6)

    def Vpp(self,V):
        Vbig = -10**6*np.ones((self.N[0]+3, self.N[1]+1, self.N[2]+1))
        Vbig[1:-1,:,:] = V
        Vpp = (Vbig[2:,:,:] - 2*Vbig[1:-1,:,:] + Vbig[:-2,:,:])/self.Delta[0]**2
        return Vpp[1:-1,:,:]

c1,c2 = parameters.c1,parameters.c2
colorFader = parameters.colorFader

"""
Set parameters
"""

"""
Following parameters taken from benchmark parameters script.
Except that bnd is changed to ensure that state constraints are satisfied.
"""

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

N_true = parameters.N_true
show_iter, show_method, show_final = 1, 1, 1
NA = parameters.NA
NA_true = parameters.NA_true
N_t = parameters.N_t
N_c = parameters.N_c
n_round_acc = parameters.n_round_acc
n_round_time = parameters.n_round_time
CT_dt_true = parameters.CT_dt_true
CT_dt_mid = parameters.CT_dt_mid
CT_dt_big = parameters.CT_dt_big
DT_dt = parameters.DT_dt

"""
Following specific to mortality script. The parameters are:

    * ages depicted (this is k_set+20);
    * (dis)utility from death;
    * parameters for health function;
    * parameters for disutility of work;
    * maximum health expenditures.
"""

N = parameters.Nexample
k_set=[0,35,55]
D=-80
Lam0low = 0.001
Lam0high = 0.05
Lam1low = 3.0
Lam1high = 0.0
etalow = 0.9
etahigh = 0.7
mbar = 1.

W = CT_nonstat_IFP_mortality(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
bnd=bnd_NS,N=(N[0],N[1],NA),maxiter=maxiter,tol=tol,show_method=show_method,
show_iter=show_iter,show_final=show_final,D=D,mbar=mbar,etalow=etalow,etahigh=etahigh,
Lam0low=Lam0low, Lam0high=Lam0high,Lam1low=Lam1low,Lam1high=Lam1high)

form='nonlinear'
V, pol, t, t_array = W.solve_seq_imp(form=form)
c, m, l = pol

for k in k_set:
    fig, ax = plt.subplots()
    for j in range(N[1]+1):
        color = colorFader(c1,c2,j/N[1])
        if j in [0,N[1]]:
            inc = W.ybar*np.round(np.exp(W.grid[1][j]),2)
            ax.plot(W.grid[0][1:], V[1:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(W.grid[0][1:], V[1:,j,k], color=color, linewidth=1)
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Value function at age {0}'.format(int(W.grid[2][k]*W.Delta[2])+20))
    destin = '../../main/figures/mortality_V_{0}.eps'.format(int(W.grid[2][k]*W.Delta[2]))
    plt.savefig(destin, format='eps', dpi=1000)
    plt.close()
    #plt.show()

    fig, ax = plt.subplots()
    for j in range(N[1]+1):
        color = colorFader(c1,c2,j/N[1])
        if j in [0,N[1]]:
            inc = W.ybar*np.round(np.exp(W.grid[1][j]),2)
            ax.plot(W.grid[0][1:], c[1:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(W.grid[0][1:], c[1:,j,k], color=color, linewidth=1)
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Consumption at age {0}'.format(int(W.grid[2][k]*W.Delta[2])+20))
    destin = '../../main/figures/mortality_c_{0}.eps'.format(int(W.grid[2][k]*W.Delta[2]))
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

    fig, ax = plt.subplots()
    for j in range(N[1]+1):
        color = colorFader(c1,c2,j/N[1])
        if j in [0,N[1]]:
            inc = W.ybar*np.round(np.exp(W.grid[1][j]),2)
            ax.plot(W.grid[0][1:], pol[1][1:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(W.grid[0][1:], pol[1][1:,j,k], color=color, linewidth=1)
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Medical expenditures at age {0}'.format(int(W.grid[2][k]*W.Delta[2])+20))
    destin = '../../main/figures/mortality_m_{0}.eps'.format(int(W.grid[2][k]*W.Delta[2]))
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

    fig, ax = plt.subplots()
    for j in range(N[1]+1):
        color = colorFader(c1,c2,j/N[1])
        if j in [0,N[1]]:
            inc = W.ybar*np.round(np.exp(W.grid[1][j]),2)
            ax.plot(W.grid[0][1:], 100*W.lam_func(pol[1][1:,j,k],k,form), color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(W.grid[0][1:], 100*W.lam_func(pol[1][1:,j,k],k,form), color=color, linewidth=1)
    #plt.xlim([0,50])
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Percent chance of death at age {0}'.format(int(W.grid[2][k]*W.Delta[2])+20))
    destin = '../../main/figures/mortality_death_{0}.eps'.format(int(W.grid[2][k]*W.Delta[2]))
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

    fig, ax = plt.subplots()
    for j in range(N[1]+1):
        color = colorFader(c1,c2,j/N[1])
        if j in [0,N[1]]:
            inc = W.ybar*np.round(np.exp(W.grid[1][j]),2)
            ax.plot(W.grid[0][1:], pol[2][1:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(W.grid[0][1:], pol[2][1:,j,k], color=color, linewidth=1)
    #plt.xlim([0,50])
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Labor at age {0}'.format(int(W.grid[2][k]*W.Delta[2])+20))
    destin = '../../main/figures/mortality_l_{0}.eps'.format(int(W.grid[2][k]*W.Delta[2]))
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()
