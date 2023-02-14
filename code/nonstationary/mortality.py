"""
Class constructors for the mortality choice section of "The Art of Temporal Approximation".

Author: Thomas Phelan.
Email: tom.phelan@clev.frb.org.

This script contains two class constructors:

    1. DT_IFP_mortality: discrete-time IFP (stationary and age-dependent)
    2. CT_nonstat_IFP_mortality: continuous-time nonstationary (age-dependent) IFP

Troubleshooting. Some probems that can arise:

    * be careful that asset bounds are chosen s.t. agent wants to dissave at
    the upper bound. Otherwise we get a "spike" in consumption.
    * notice that FOCs might not characterize optimum EVEN IF value function is
    concave, because RHS of Bellman equation is not concave in c and m.

This last observation is a point in favor of the continuous-time framework,
because the continuous-time framework is unaffected by these subtleties.
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

class DT_IFP_mortality(object):
    def __init__(self, rho=0.05, r=0.02, gamma=2, ybar=1, mubar=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, bnd=[[0,60],[-0.8,0.8]], N=[100,10],
    NA=60, N_t = 10, N_c=2000, dt=0.01, tol=10**-6, maxiter=400, maxiter_PFI=25,
    show_method=1, show_iter=1, show_final=1, kappac=2, D=-40, Lam0=0.5, Lam1=0.5,
    mbar = 0.2, mfix = 0.2):
        self.rho, self.r, self.gamma = rho, r, gamma
        self.ybar, self.mubar, self.sigma = ybar, mubar, sigma
        self.N, self.bnd, self.NA, self.N_t = N, bnd, NA, N_t
        self.tol, self.maxiter, self.maxiter_PFI = tol, maxiter, maxiter_PFI
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(2)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(2)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],indexing='ij')
        self.ii, self.jj = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),indexing='ij')
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1] - 2) #vanish on boundaries
        self.dt, self.Dt = dt, dt + 0*self.xx[0]
        self.dt_small = self.dt/self.N_t
        self.grid_A = np.linspace(0+self.dt,self.dt*(self.NA-1),self.NA-1)
        self.N_c, self.M = N_c, (self.N[0]-1)*(self.N[1]-1)
        #following is "zero net saving"
        self.c0 = self.r*self.xx[0]/(1+self.dt*self.r) + self.ybar*np.exp(self.xx[1])
        self.kappac = kappac
        self.cmax = self.kappac*self.c0
        self.show_method, self.show_iter, self.show_final = show_method, show_iter, show_final
        self.iter_keys = list(itertools.product(range(self.N[0]-1), range(self.N[1]-1)))
        self.probs, self.p_z = {}, {}
        self.p_z['KD'] = self.KD()
        self.p_z['Tauchen'] = self.Tauchen()
        self.check = np.min(self.p_z['KD']) > -10**-8
        for disc in ['KD','Tauchen']:
            self.probs[disc] = self.p_func_diff((self.ii, self.jj), prob=disc)
        self.first_Vp = self.Vp_cont_jit(self.c0,'KD')
        #additional parameters relevant for mortality choice
        self.Lam0, self.Lam1, self.D, self.mbar, self.mfix = Lam0, Lam1, D, mbar, mfix
        self.clow = (self.grid[0][self.ii] - self.grid[0][-1]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj]) - self.mfix + 10**-4
        self.chigh = (self.grid[0][self.ii] - self.grid[0][0]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj]) - self.mfix - 10**-4

    def lam_func(self,m):
        return self.Lam0*np.exp(-self.Lam1*m)

    def H(self,m):
        return -self.Lam1*self.Lam0*np.exp(-self.Lam1*m)/((1 + self.dt*self.r)*(1 - self.Lam0*np.exp(-self.Lam1*m)*self.dt))

    def H_inv(self,X):
        arg = np.maximum(self.Lam0*self.dt - self.Lam1*self.Lam0*X**(-1)/(1 + self.dt*self.r), 10**(-10))
        return self.Lam1**(-1)*np.log(arg)

    def cbound_func(self,m):
        clow = (self.grid[0][self.ii] - self.grid[0][-1]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj]) - m + 10**-4
        chigh = (self.grid[0][self.ii] - self.grid[0][0]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj]) - m - 10**-4
        return clow, chigh

    def u(self,c):
        if self.gamma==1:
            return np.log(c)
        else:
            return c**(1-self.gamma)/(1-self.gamma)

    def u_prime_inv(self,x):
        return x**(-1/self.gamma)

    def p_func_diff(self,ind,prob='KD'):
        ii,jj = ind
        p_func_diff = {}
        P = self.p_z[prob]
        for k in range(self.N[1]-1):
            p_func_diff[k] = P[ii,jj,k]
        return p_func_diff

    def Tauchen(self):
        p = np.zeros((self.N[0]-1,self.N[1]-1,self.N[1]-1))
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),range(1,self.N[1]-2),indexing='ij')
        up_bnd = (self.grid[1][kkk] + self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(np.sqrt(self.dt)*self.sigma)
        down_bnd = (self.grid[1][kkk] - self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(np.sqrt(self.dt)*self.sigma)
        p[iii,jjj,kkk] = norm.cdf(up_bnd) - norm.cdf(down_bnd)
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),0,indexing='ij')
        up_bnd = (self.grid[1][kkk] + self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(np.sqrt(self.dt)*self.sigma)
        p[iii,jjj,kkk] = norm.cdf(up_bnd)
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),self.N[1]-2,indexing='ij')
        down_bnd = (self.grid[1][kkk] - self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(np.sqrt(self.dt)*self.sigma)
        p[iii,jjj,kkk] = 1 - norm.cdf(down_bnd)
        return p

    def KD_z(self):
        p_z_small = np.zeros((self.N[1]-1,self.N[1]-1))
        sig = self.sigma + 0*self.grid[1]
        sig[0], sig[-1] = 0, 0
        pup_z_small = self.dt_small*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-self.grid[1]),0))/self.Delta[1]**2
        pdown_z_small = self.dt_small*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(self.grid[1]),0))/self.Delta[1]**2
        p_z_small[range(self.N[1]-1),range(self.N[1]-1)] = 1 - pup_z_small - pdown_z_small
        p_z_small[range(self.N[1]-2),range(1,self.N[1]-1)] = pup_z_small[:-1]
        p_z_small[range(1,self.N[1]-1),range(self.N[1]-2)] = pdown_z_small[1:]
        p_z_T = np.linalg.matrix_power(np.mat(p_z_small).T,self.N_t)
        return np.array(p_z_T.T)

    def KD(self):
        p = np.zeros((self.N[0]-1,self.N[1]-1,self.N[1]-1))
        p_z = self.KD_z()
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),range(self.N[1]-1),indexing='ij')
        p[iii,jjj,kkk] = p_z[jjj,kkk]
        return p

    def m(self,V_slice,prob):
        Vp_cont = self.Vp_cont_jit(V_slice,prob)
        V_cont = self.V_p_jit(V_slice,self.p_z[prob],self.N)
        X = Vp_cont/(self.D - V_cont)
        m = np.minimum(np.maximum(self.H_inv(X),0), self.mbar)
        m[self.D > V_cont] = 0
        m[0,:] = 0
        return m

    def G(self,m,V_slice,prob):
        Vp_cont = self.Vp_cont_jit(V_slice,prob)
        return (1+self.dt*self.r)*(1-self.lam_func(m)*self.dt)*np.exp(-self.rho*self.dt)*Vp_cont

    #only EGM used. V_slice in the following is FUTURE V.
    def polupdate(self,V_slice,prob):
        cnew = np.zeros((self.N[0]-1,self.N[1]-1))
        mnew = self.m(V_slice,prob)
        G = self.G(mnew,V_slice,prob)
        bb, zz = self.grid[0][self.ii], self.grid[1][self.jj]
        cand_b = self.dt*(self.u_prime_inv(G) + mnew - self.ybar*np.exp(zz)) + bb/(1+self.dt*self.r)
        cand_c = (cand_b - bb/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(zz) - mnew
        for j in range(self.N[1]-1):
            cnew[:,j] = interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(self.grid[0])
        cbound = self.cbound_func(mnew)
        return np.minimum(np.maximum(cnew, cbound[0]), cbound[1]), mnew

    #following is rapid construction of continuation values for V' or V
    #remember numba cannot call any self. attributes.
    @staticmethod
    @jit(nopython=True)
    def V_p_jit(V,p,N):
        V_cont = np.zeros((N[0]-1,N[1]-1))
        #k loops over future income, i over assets, and j over current income
        for k in range(N[1]-1):
            for i in range(N[0]-1):
                for j in range(N[1]-1):
                    V_cont[i,j] += p[i,j,k]*V[i,k]
        return V_cont

    def Vp_cont_jit(self,V,prob):
        Vp = np.zeros((self.N[0]-1,self.N[1]-1))
        Vp[1:-1,:] = (V[2:,:] - V[:-2,:])/(2*self.Delta[0])
        Vp[0,:], Vp[-1,:] = Vp[1,:], Vp[-2,:]
        return self.V_p_jit(Vp,self.p_z[prob],self.N)

    def weights_indices(self,b_prime):
        #10**-8 term in following due to rounding concern. subtract boundary
        #point from argument (to write b as multiple of Delta_a) and subtract 1.
        ind = np.array([np.floor((b-self.bnd[0][0])/self.Delta[0] + 10**-8)-1 for b in b_prime]).astype(int)
        return 1 - (b_prime - (self.grid[0][0]+ind*self.Delta[0]))/self.Delta[0], ind

    #often faster to NOT build matrix and instead write out algebra as follows
    def Jacobi_no_matrix(self,pol,V,prob):
        PV = np.zeros((self.N[0]-1,self.N[1]-1))
        c, m = pol
        b_prime = (1 + self.dt*self.r)*(self.xx[0] + self.dt*(self.ybar*np.exp(self.xx[1]) - c - m))
        alpha, ind = self.weights_indices(b_prime)
        ii, jj = self.ii, self.jj
        for key in range(self.N[1]-1):
            #lower then upper weights on asset transition:
            row, col = ii*(self.N[1]-1) + jj, ind[ii,jj]*(self.N[1]-1) + jj + key
            ind_bnd = (col>-1)*(col<self.M)
            ii_, jj_ = ii[ind_bnd], jj[ind_bnd]
            ind_ = ind[ii_,jj_]
            alpha_, prob_ = alpha[ii_,jj_], self.probs[prob][key][ii_,jj_]
            PV[ii_,jj_] = PV[ii_,jj_] + alpha_*prob_*V[ind_, key] + (1-alpha_)*prob_*V[ind_+1, key]
        future = (1 - self.lam_func(m)*self.dt)*PV + self.lam_func(m)*self.dt*self.D
        return self.dt*self.u(c) + np.exp(-self.rho*self.dt)*future

    def nonstat_solve(self,prob='KD',matrix=False):
        if self.show_method==1:
            print("Starting non-stationary problem with {0} probabilities".format(prob))
        V = np.zeros((self.N[0]-1,self.N[1]-1,self.NA-1))
        c = np.zeros((self.N[0]-1,self.N[1]-1,self.NA-1))
        m = np.zeros((self.N[0]-1,self.N[1]-1,self.NA-1))
        time_array = np.zeros((2,self.NA-1))
        tic = time.time()
        c[:,:,self.NA-2] = self.ybar*np.exp(self.xx[1]) + self.xx[0]/self.dt
        m[:,:,self.NA-2] = np.zeros((self.N[0]-1,self.N[1]-1))
        V[:,:,self.NA-2] = self.Jacobi_no_matrix((c[:,:,self.NA-2],m[:,:,self.NA-2]),np.zeros((self.N[0]-1,self.N[1]-1)),prob)
        toc = time.time()
        print("Initialize Discrete-time problem")
        tic = time.time()
        for k in range(self.NA-2):
            if (int(self.NA-2-k-1) % 20 ==0):
                print("Age:",self.NA-2-k-1)
            V_imp = V[:,:,self.NA-2-k]
            print("Concave?", self.Vpp_check(V_imp))
            tc1 = time.time()
            new_pol = self.polupdate(V_imp,prob)
            c[:,:,self.NA-2-k-1] = new_pol[0]
            m[:,:,self.NA-2-k-1] = new_pol[1]
            tc2 = time.time()
            tV1 = time.time()
            V[:,:,self.NA-2-k-1] = self.Jacobi_no_matrix(new_pol,V_imp,prob)
            tV2 = time.time()
            time_array[:,self.NA-2-k-1] = tc2-tc1, tV2-tV1
        toc = time.time()
        if self.show_final==1:
            print("Time taken:", toc-tic)
        return V, (c,m), toc-tic, time_array

    def P_func(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M)],B[(B>-1)*(B<self.M)],C[(B>-1)*(B<self.M)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def Vpp_check(self,V_slice):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]-1))
        Vbig[1:-1,:] = V_slice
        Vpp = (Vbig[2:,:] - 2*Vbig[1:-1,:] + Vbig[:-2,:])/self.Delta[0]**2
        return np.max(Vpp)<0

    def Vpp(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]-1, self.NA-1))
        Vbig[1:-1,:,:] = V
        Vpp = (Vbig[2:,:,:] - 2*Vbig[1:-1,:,:] + Vbig[:-2,:,:])/self.Delta[0]**2
        return Vpp[1:-1,:,:]

"""
Non-stationary continuous-time income fluctuation problem with mortality choice.
"""

class CT_nonstat_IFP_mortality(object):
    def __init__(self, rho=0.05, r=0.03, gamma=2., ybar=1, mubar=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, bnd=[[0,60],[-0.6,0.6],[0,60]], N=(40,10,10),
    dt = 10**(-4), tol=10**-6, maxiter=200, maxiter_PFI=25, mono_tol=10**(-6),
    show_method=1, show_iter=1, show_final=1, kappac=2, D=-40, Lam0=0.5, Lam1=0.5,
    mbar = 4, mfix = 0.2):
        self.r, self.rho, self.gamma = r, rho, gamma
        self.ybar, self.mubar, self.sigma = ybar, mubar, sigma
        self.tol, self.mono_tol = tol, mono_tol
        self.maxiter, self.maxiter_PFI = maxiter, maxiter_PFI
        self.N, self.M, self.M_slice = N, (N[0]-1)*(N[1]-1)*(N[2]-1), (N[0]-1)*(N[1]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(3)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        self.ii, self.jj, self.kk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),range(self.N[2]-1), indexing='ij')
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1] - 2)
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1)]
        self.trans_keys_slice = [(1,0),(-1,0),(0,1),(0,-1)]
        self.c0 = self.r*self.xx[0] + self.ybar*np.exp(self.xx[1])
        self.dt, self.Dt = dt, dt + 0*self.c0
        self.kappac = kappac
        self.cmax = 1.5*self.xx[0][-1] + 0*self.c0
        self.show_method, self.show_iter, self.show_final = show_method, show_iter, show_final
        #additional parameters relevant for death
        self.mfix = mfix
        self.Lam0, self.Lam1, self.D, self.mbar = Lam0, Lam1, D, mbar

    def lam_func(self,m):
        return self.Lam0*np.exp(-self.Lam1*m)

    def p_func_slice(self,ind,pol):
        ii,jj = ind
        p_func_diff = {}
        x = (self.xx[0][ii,jj,0],self.xx[1][ii,jj,0])
        dt, sig = self.Dt[ii,jj,0], self.sigsig[ii,jj,0]
        c, m = pol[0][ii,jj], pol[1][ii,jj]
        denom = 1 - self.lam_func(m)*dt
        p_func_diff[(1,0)] = (dt/self.Delta[0])*np.maximum(self.r*x[0]+np.exp(x[1])-c,0)/denom
        p_func_diff[(-1,0)] = (dt/self.Delta[0])*(np.maximum(-(self.r*x[0]+np.exp(x[1])-c),0)+m)/denom
        p_func_diff[(0,1)] = (dt/self.Delta[1]**2)*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-x[1]),0))/denom
        p_func_diff[(0,-1)] = (dt/self.Delta[1]**2)*(sig**2/2 + self.Delta[1]*np.maximum(-self.mubar*(-x[1]),0))/denom
        p_func_diff[(0,0,1)] = (dt/self.Delta[2])/denom
        return p_func_diff

    #transition matrix CONDITIONAL on survival.
    def P_slice(self,pol):
        ii, jj = np.meshgrid(range(self.N[0]-1), range(self.N[1]-1), indexing='ij')
        row = ii*(self.N[1]-1) + jj
        diag = 1 - sum(self.p_func_slice((ii,jj),pol).values())
        P = self.P_func_slice(row,row,diag)
        for key in self.trans_keys_slice:
            column = row + key[0]*(self.N[1]-1) + key[1]
            P = P + self.P_func_slice(row,column,self.p_func_slice((ii,jj),pol)[key])
        return P

    #pol_slice = c_slice,m_slice. V_imp_up is V at higher age conditional on no death.
    def V_imp(self,V_imp_up,pol_slice):
        Dt_slice = self.Dt[:,:,0]
        c_slice, m_slice = pol_slice
        b = Dt_slice*c_slice**(1-self.gamma)/(1-self.gamma) \
        + np.exp(-self.rho*Dt_slice)*Dt_slice*(self.lam_func(m_slice)*self.D + V_imp_up/self.Delta[2])
        D = (np.exp(-self.rho*Dt_slice)*(1 - self.lam_func(m_slice)*Dt_slice)).reshape((self.M_slice,))
        B = sp.eye(self.M_slice) - sp.diags(D)*self.P_slice(pol_slice)
        b, B = b/self.dt, B/self.dt
        return sp.linalg.spsolve(B, b.reshape((self.M_slice,))).reshape((self.N[0]-1,self.N[1]-1))

    def polupdate_slice(self,V_imp):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1))
        Vbig[1:-1,1:-1] = V_imp
        VB1 = (Vbig[1:-1,1:-1]-Vbig[:-2,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1]-Vbig[1:-1,1:-1])/self.Delta[0]
        #medical expenditures:
        m_star = -np.log(VB1*(np.maximum(V_imp - self.D, 10**(-10)))**(-1)/(self.Lam0*self.Lam1))/self.Lam1
        m = np.maximum(0,np.minimum(m_star,self.mbar))
        c0, cmax, Dt_slice = self.c0[:,:,0], self.cmax[:,:,0], self.Dt[:,:,0]
        obj = lambda c: c**(1-self.gamma)/(1-self.gamma) + np.exp(-self.rho*Dt_slice) \
        *(np.maximum(c0 - c, 0)*VF1 - np.maximum(-(c0 - c), 0)*VB1)
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum((np.exp(-self.rho*Dt_slice)*VF1)**(-1/self.gamma), c0)
            chigh = np.maximum((np.exp(-self.rho*Dt_slice)*VB1)**(-1/self.gamma), c0)
        clow[VF1<=0], chigh[VB1<=0] = cmax[VF1<=0], cmax[VB1<=0]
        runmax = np.concatenate((obj(c0).reshape(1,self.M_slice), \
        obj(clow).reshape(1,self.M_slice), obj(chigh).reshape(1,self.M_slice)))
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]-1,self.N[1]-1)
        C = (IND==0)*c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:] = np.minimum(C[0,:], c0[0,:])
        C[-1,:] = np.maximum(C[-1,:], c0[-1,:])
        #two-point case:
        #m_ind = (self.lam_func(self.mbar)*(self.D - V_imp) - self.mbar*VB1 > self.lam_func(0)*(self.D - V_imp))
        #m = self.mbar*m_ind
        return C, m

    def solveVslice(self,V_imp_up,pol_guess):
        V = self.V_imp(V_imp_up,pol_guess)
        eps, i = 1, 1
        while i < 20 and eps > self.tol:
            V1 = self.V_imp(V_imp_up,self.polupdate_slice(V))
            eps = np.amax(np.abs(V - V1))
            V, i = V1, i+1
        return V

    def solve_seq_imp(self):
        if self.show_method==1:
            print("Starting sequential PFI")
        V = np.zeros((self.N[0]-1,self.N[1]-1,self.N[2]-1))
        c = np.zeros((self.N[0]-1,self.N[1]-1,self.N[2]-1))
        m = np.zeros((self.N[0]-1,self.N[1]-1,self.N[2]-1))
        time_array = np.zeros((2,self.N[2]-1))
        pol_guess = (self.c0[:,:,0], 0*self.c0[:,:,0])
        V[:,:,self.N[2]-2] = self.solveVslice(np.zeros((self.N[0]-1,self.N[1]-1)), pol_guess)
        c[:,:,self.N[2]-2], m[:,:,self.N[2]-2] = self.polupdate_slice(V[:,:,self.N[2]-2])
        tic = time.time()
        for k in range(self.N[2]-2):
            if (int(self.N[2]-2-k-1) % 20 == 0):
                print("Age:", self.N[2]-2-k-1)
            tV1 = time.time()
            pol = (c[:,:,self.N[2]-2-k], m[:,:,self.N[2]-2-k])
            V[:,:,self.N[2]-2-k-1] = self.solveVslice(V[:,:,self.N[2]-2-k],pol)
            print("Concave?", self.Vpp_check(V[:,:,self.N[2]-2-k-1]))
            tV2 = time.time()
            tc1 = time.time()
            new_pol = self.polupdate_slice(V[:,:,self.N[2]-2-k-1])
            c[:,:,self.N[2]-2-k-1] = new_pol[0]
            m[:,:,self.N[2]-2-k-1] = new_pol[1]
            tc2 = time.time()
            time_array[:,self.N[2]-2-k-1] = tc2-tc1, tV2-tV1
        toc = time.time()
        if self.show_final==1:
            print("Time taken:", toc-tic)
        return V, (c,m), toc-tic, time_array

    def P_func_slice(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M_slice)],B[(B>-1)*(B<self.M_slice)],C[(B>-1)*(B<self.M_slice)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M_slice,self.M_slice))

    def P_func(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M)],B[(B>-1)*(B<self.M)],C[(B>-1)*(B<self.M)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

    def Vpp_check(self,V_slice):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]-1))
        Vbig[1:-1,:] = V_slice
        Vpp = (Vbig[2:,:] - 2*Vbig[1:-1,:] + Vbig[:-2,:])/self.Delta[0]**2
        return np.max(Vpp)<0

    def Vpp(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]-1, self.N[2]-1))
        Vbig[1:-1,:,:] = V
        Vpp = (Vbig[2:,:,:] - 2*Vbig[1:-1,:,:] + Vbig[:-2,:,:])/self.Delta[0]**2
        return Vpp[1:-1,:,:]

c1,c2 = parameters.c1,parameters.c2
colorFader = parameters.colorFader

"""
Set parameters
"""

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

N_true, N_c = parameters.N_true, parameters.N_c
show_iter, show_method, show_final = 1, 1, 1
N_true, N_c = parameters.N_true, parameters.N_c
n_round_acc = parameters.n_round_acc
n_round_time = parameters.n_round_time
CT_dt_true = parameters.CT_dt_true
CT_dt_mid = parameters.CT_dt_mid
CT_dt_big = parameters.CT_dt_big
DT_dt = parameters.DT_dt
NA = parameters.NA

D=-20
Lam1=10
Lam0=0.1
mfix=0.2
mbar=4.0

N = (200,10)
X = DT_IFP_mortality(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
N=N,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
show_method=show_method,show_iter=show_iter,show_final=show_final,dt=1,NA=NA,
D=D,Lam1=Lam1,Lam0=Lam0,mfix=mfix,mbar=mbar)
W = CT_nonstat_IFP_mortality(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,bnd=bnd_NS,
N=(N[0],N[1],NA),maxiter=maxiter,tol=tol,show_method=show_method,
show_iter=show_iter,show_final=show_final,D=D,Lam1=Lam1,Lam0=Lam0,mfix=mfix,mbar=mbar)

V, pol, t, t_array = {}, {}, {}, {}
c, m, Vpp = {}, {}, {}
V[X], pol[X], t[X], t_array[X] = X.nonstat_solve()
V[W], pol[W], t[W], t_array[W] = W.solve_seq_imp()

for x in [X,W]:
    c[x], m[x] = pol[x]
    Vpp[x] = x.Vpp(V[x])

dc = pol[X][0] - pol[W][0]
dm = pol[X][1] - pol[W][1]

"""
We can now look at the value function for the discrete-time case
"""

k_list = [0]
k_dict = {0:'initial',-1:'final'}
class_dict = {W:'continuous-time',X:'discrete-time'}
class_list = [X,W]

"""
Initial period only
"""

k=0

"""
Value functions
"""

for x in class_list:
    fig, ax = plt.subplots()
    for j in range(N[1]-1):
        color = colorFader(c1,c2,j/(N[1]-1))
        if j in [0,N[1]-2]:
            inc = x.ybar*np.round(np.exp(x.grid[1][j]),2)
            ax.plot(x.grid[0], V[x][:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(x.grid[0], V[x][:,j,k], color=color, linewidth=1)
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Value function in {0} period'.format(k_dict[k]))
    destin = '../../main/figures/mortality_V_{0}.eps'.format(class_dict[x])
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

"""
Consumption functions
"""

for x in class_list:
    fig, ax = plt.subplots()
    for j in range(N[1]-1):
        color = colorFader(c1,c2,j/(N[1]-1))
        if j in [0,N[1]-2]:
            inc = x.ybar*np.round(np.exp(x.grid[1][j]),2)
            ax.plot(x.grid[0], c[x][:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(x.grid[0], c[x][:,j,k], color=color, linewidth=1)
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Consumption in {0} period'.format(k_dict[k]))
    destin = '../../main/figures/mortality_c_{0}.eps'.format(class_dict[x])
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

"""
Medical expenditure functions
"""

for x in class_list:
    fig, ax = plt.subplots()
    for j in range(N[1]-1):
        color = colorFader(c1,c2,j/(N[1]-1))
        if j in [0,N[1]-2]:
            inc = x.ybar*np.round(np.exp(x.grid[1][j]),2)
            ax.plot(x.grid[0], pol[x][1][:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
        else:
            ax.plot(x.grid[0], pol[x][1][:,j,k], color=color, linewidth=1)
    plt.xlabel('Assets $b$')
    plt.legend()
    plt.title('Medical expenditures in {0} period'.format(k_dict[k]))
    destin = '../../main/figures/mortality_m_{0}.eps'.format(class_dict[x])
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

"""
Differences across methods
"""

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = x.ybar*np.round(np.exp(x.grid[1][j]),2)
        ax.plot(x.grid[0], V[X][:,j,k] - V[W][:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(x.grid[0], V[X][:,j,k] - V[W][:,j,k], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Value function differences in {0} period'.format(k_dict[k]))
destin = '../../main/figures/mortality_V_diff.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = x.ybar*np.round(np.exp(x.grid[1][j]),2)
        ax.plot(x.grid[0], c[X][:,j,k] - c[W][:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(x.grid[0], c[X][:,j,k] - c[W][:,j,k], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Consumption differences in {0} period'.format(k_dict[k]))
destin = '../../main/figures/mortality_c_diff.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = x.ybar*np.round(np.exp(x.grid[1][j]),2)
        ax.plot(x.grid[0], m[X][:,j,k] - m[W][:,j,k], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(x.grid[0], m[X][:,j,k] - m[W][:,j,k], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Medical expenditures differences in {0} period'.format(k_dict[k]))
destin = '../../main/figures/mortality_m_diff.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
