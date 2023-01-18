"""
Class constructors for "The Art of Temporal Approximation".

Author: Thomas Phelan.
Email: tom.phelan@clev.frb.org.

This script contains three class constructors:

    1. DT_IFP: discrete-time IFP (stationary and age-dependent)
    2. CT_stat_IFP: continuous-time stationary IFP
    3. CT_nonstat_IFP: continuous-time nonstationary (age-dependent) IFP

Some miscellaneous notes:

    * All "solve" methods return a quadruple V, c, time, i.
    * Timestep is constant in every class. The theory allows state-dependence
    but the paper does not exploit this. dt is a scalar; Dt is a grid function.
    * In discrete-time setting max_age is inferred from dt (which equals dA)
    * In continuous-time setting max age is a primitive, as bnd enters explicitly.
    * In implementation of EGM in DT_IFP, we replace V_prime with small positive
    number if negative. This never appears to be chosen in the optimal policy.
    * No need for fill_value="extrapolate" in interp1d in DT_IFP. If Python wants
    to extrapolate then something else has gone wrong, because consumption should
    be restricted so that future assets lie on exogenous grid.
    * To address potential overflow problems when using sparse solver, we divide
    both sides of the system by Dt to be commensurate with flow utility.
    * Boundary points of bnd are excluded. e.g. if bnd[0] = [0, 60] and N[0] = 100,
    the grid for assets has 99 points between Delta_a and 60 - Delta_a inclusive.
    * In the DT_IFP and CT_IFP cases there is no need for an arbitrary ind
    argument, because these only ever take ii,jj = self.ii, self.jj. However, this
    generality is necessary in the nonstationary problem, where "slices" arise.
    * The DT_IFP class constructor allows for both the same probability
    transitions in the continuous-time case, denoted prob='KD' for "Kushner and
    Dupuis", and also the method of Tauchen (1986).
    * All class constructors write self.cmax = self.kappac*self.c0, a vector
    that is a multiple of zero net saving. This default is 2.
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
from scipy.sparse import linalg
from scipy.sparse import diags
from scipy.interpolate import interp1d
import itertools, time

class DT_IFP(object):
    def __init__(self, rho=0.05, r=0.02, gamma=2, ybar=1, mubar=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, bnd=[[0,60],[-0.8,0.8]], N=[100,10],
    NA=60, N_c=2000, dt=0.01, tol=10**-6, maxiter=400, maxiter_PFI=25,
    show_method=1, show_iter=1, show_final=1, kappac=2):
        self.rho, self.r, self.gamma = rho, r, gamma
        self.ybar, self.mubar, self.sigma = ybar, mubar, sigma
        self.N, self.bnd, self.NA = N, bnd, NA
        self.tol, self.maxiter, self.maxiter_PFI = tol, maxiter, maxiter_PFI
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(2)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(2)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],indexing='ij')
        self.ii, self.jj = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),indexing='ij')
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1] - 2)
        self.dt, self.Dt = dt, dt + 0*self.xx[0]
        self.grid_A = np.linspace(0+self.dt,self.dt*(self.NA-1),self.NA-1)
        self.N_c, self.M = N_c, (self.N[0]-1)*(self.N[1]-1)
        #following is "zero net saving" and value of this policy
        self.c0 = self.r*self.xx[0]/(1+self.dt*self.r) + self.ybar*np.exp(self.xx[1])
        self.dt, self.Dt = dt, dt + 0*self.c0
        self.kappac = kappac
        self.cmax = self.kappac*self.c0
        #self.V0, self.V0_T = self.V(self.c0), self.V(self.c0,'Tauchen')
        self.V0 = self.V(self.c0)
        self.pup = self.dt*(self.sigsig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-self.grid[1]),0))/self.Delta[1]**2
        self.pdown = self.dt*(self.sigsig**2/2 + self.Delta[1]*np.maximum(self.mubar*(self.grid[1]),0))/self.Delta[1]**2
        self.pstay = 1 - self.pup - self.pdown
        self.prob_check = np.min(self.pstay)
        self.show_method, self.show_iter, self.show_final = show_method, show_iter, show_final
        self.iter_keys = list(itertools.product(range(self.N[0]-1), range(self.N[1]-1)))

    def u(self,c):
        if self.gamma==1:
            return np.log(c)
        else:
            return c**(1-self.gamma)/(1-self.gamma)

    def u_prime_inv(self,x):
        return x**(-1/self.gamma)

    def p_func(self,ind,prob='KD'):
        ii,jj = ind
        p_func = {}
        x = (self.xx[0][ii,jj],self.xx[1][ii,jj])
        sig = self.sigsig[ii,jj]
        if prob=='KD':
            p_func[1] = self.dt*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-x[1]),0))/self.Delta[1]**2
            p_func[-1] = self.dt*(sig**2/2 + self.Delta[1]*np.maximum(-self.mubar*(-x[1]),0))/self.Delta[1]**2
            p_func[0] = 1 - p_func[-1] - p_func[1]
            return p_func
        else:
            P = self.Tauchen()
            for k in range(self.N[1]-1):
                p_func[k] = P[ii,jj,k]
            return p_func

    def Tauchen(self):
        p = np.zeros((self.N[0]-1,self.N[1]-1,self.N[1]-1))
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),range(1,self.N[1]-2),indexing='ij')
        up_bnd = (self.grid[1][kkk] + self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(self.dt*self.sigma)
        down_bnd = (self.grid[1][kkk] - self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(self.dt*self.sigma)
        p[iii,jjj,kkk] = norm.cdf(up_bnd) - norm.cdf(down_bnd)
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),0,indexing='ij')
        up_bnd = (self.grid[1][kkk] + self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(self.dt*self.sigma)
        p[iii,jjj,kkk] = norm.cdf(up_bnd)
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),self.N[1]-2,indexing='ij')
        down_bnd = (self.grid[1][kkk] - self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(self.dt*self.sigma)
        p[iii,jjj,kkk] = 1 - norm.cdf(down_bnd)
        return p

    def KD(self):
        p = np.zeros((self.N[0]-1,self.N[1]-1,self.N[1]-1))
        pup, pdown, pstay = self.pup, self.pdown, 1 - self.pup - self.pdown
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),range(self.N[1]-1),indexing='ij')
        p[iii,jjj,jjj] = pstay[jjj,jjj]
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(self.N[1]-2),range(self.N[1]-1),indexing='ij')
        p[iii,jjj,jjj+1] = pup[jjj,jjj]
        iii, jjj, kkk = np.meshgrid(range(self.N[0]-1),range(1,self.N[1]-1),range(self.N[1]-1),indexing='ij')
        p[iii,jjj,jjj-1] = pdown[jjj,jjj]
        return p

    def P(self,c,prob='KD'):
        P = sp.coo_matrix((self.M,self.M))
        b_prime = (1 + self.dt*self.r)*(self.xx[0] + self.dt*(self.ybar*np.exp(self.xx[1]) - c))
        alpha, ind = self.weights_indices(b_prime)
        ii, jj = self.ii, self.jj
        if prob=='KD':
            for key in [-1,0,1]:
                #lower then upper weights on asset transition:
                row, col = ii*(self.N[1]-1) + jj, ind[ii,jj]*(self.N[1]-1) + jj + key
                P = P + self.P_func(row,col,alpha[ii,jj]*self.p_func((ii,jj),prob)[key])
                P = P + self.P_func(row,col+(self.N[1]-1),(1-alpha[ii,jj])*self.p_func((ii,jj),prob)[key])
            return P
        else:
            for key in range(self.N[1]-1):
                #lower then upper weights on asset transition:
                row, col = ii*(self.N[1]-1) + jj, ind[ii,jj]*(self.N[1]-1) + key
                P = P + self.P_func(row,col,alpha[ii,jj]*self.p_func((ii,jj),prob)[key])
                P = P + self.P_func(row,col+(self.N[1]-1),(1-alpha[ii,jj])*self.p_func((ii,jj),prob)[key])
            return P

    def polupdate(self,method,V,prob,stat=1):
        if method=='BF':
            V_cont = np.zeros((self.N[0]-1,self.N[1]-1,self.N_c))
            ii_, jj_, kk_ = np.meshgrid(range(self.N[0]-1),range(self.N[1]-1),range(self.N_c),indexing='ij')
            clow = (self.grid[0][self.ii] - self.grid[0][-1]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj]) + 10**-4
            chigh = (self.grid[0][self.ii] - self.grid[0][0]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj]) - 10**-4
            cgrid_rest = np.linspace(np.maximum(clow,10**-8), np.minimum(chigh, np.max(self.cmax)), self.N_c)
            cgrid_rest = np.transpose(cgrid_rest, axes=[1, 2, 0])
            b_prime = (1 + self.dt*self.r)*(self.grid[0][ii_] + self.dt*(self.ybar*np.exp(self.grid[1][jj_]) - cgrid_rest))
            V_interp = [interp1d(self.grid[0], V[:,k])(b_prime) for k in range(self.N[1]-1)]
            if prob=='KD':
                p_z = self.KD()
            else:
                p_z = self.Tauchen()
            for k in range(self.N[1]-1):
                V_cont = V_cont + p_z[ii_,jj_,k]*V_interp[k]
            RHS = self.dt*self.u(cgrid_rest) + np.exp(-self.rho*self.dt)*V_cont
            cnew = np.array([cgrid_rest[key[0],key[1],np.argmin(-RHS[key[0],key[1],:])] for key in self.iter_keys])
            cnew = cnew.reshape((self.N[0]-1,self.N[1]-1))
        else:
            c_new = np.zeros((self.N[0]-1,self.N[1]-1))
            cand_b = np.zeros((self.N[0]-1,self.N[1]-1))
            cand_c = np.zeros((self.N[0]-1,self.N[1]-1))
            Vp_cont = self.Vp_cont(V,prob)
            #candidate asset values and consumption
            cand_b[self.ii,self.jj] = self.dt*self.u_prime_inv((1+self.dt*self.r) \
            *np.exp(-self.rho*self.dt)*Vp_cont[self.ii,self.jj]) \
            + self.grid[0][self.ii]/(1+self.dt*self.r) - self.dt*self.ybar*np.exp(self.grid[1][self.jj])
            cand_c[self.ii,self.jj] = (cand_b[self.ii,self.jj] \
            - self.grid[0][self.ii]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj])
            for j in range(self.N[1]-1):
                c_new[:,j] = interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(self.grid[0])
            return self.feasible_c(c_new)
        return cnew

    def Vp_cont(self,V,prob):
        Vp_cont, Vp = np.zeros((self.N[0]-1,self.N[1]-1)), np.zeros((self.N[0]-1,self.N[1]-1))
        Vp[1:-1,:] = (V[2:,:] - V[:-2,:])/(2*self.Delta[0])
        Vp[0,:], Vp[-1,:] = Vp[1,:], Vp[-2,:]
        if prob=='KD':
            pup, pdown, pstay = self.pup, self.pdown, 1 - self.pup - self.pdown
            #Vp for interior z, z=z_low, and z=z_high, respectively:
            ii_int, jj_int = np.meshgrid(range(self.N[0]-1),range(1,self.N[1]-2),indexing='ij')
            Vp_cont[ii_int, jj_int] = pdown[ii_int,jj_int]*Vp[ii_int,jj_int-1] \
            + pup[ii_int,jj_int]*Vp[ii_int,jj_int+1] + pstay[ii_int,jj_int]*Vp[ii_int,jj_int]
            ii_low, jj_low = np.meshgrid(range(self.N[0]-1),0,indexing='ij')
            Vp_cont[ii_low, jj_low] = pup[ii_low,jj_low]*Vp[ii_low,jj_low+1] \
            + pstay[ii_low,jj_low]*Vp[ii_low,jj_low]
            ii_high, jj_high = np.meshgrid(range(self.N[0]-1),self.N[1]-2,indexing='ij')
            Vp_cont[ii_high, jj_high] = pdown[ii_high,jj_high]*Vp[ii_high,jj_high-1] \
            + pstay[ii_high,jj_high]*Vp[ii_high,jj_high]
            #replace negative values with small positive number
            Vp_cont[Vp_cont[self.ii,self.jj] <= 10**-3] = 10**-3
        else:
            Vp_cont = self.Vp_cont_Tauchen(V)
        return Vp_cont

    def Vp_cont_Tauchen(self,V):
        p_z = self.Tauchen()
        Vp_cont, Vp = np.zeros((self.N[0]-1,self.N[1]-1)), np.zeros((self.N[0]-1,self.N[1]-1))
        Vp[1:-1,:] = (V[2:,:] - V[:-2,:])/(2*self.Delta[0])
        Vp[0,:], Vp[-1,:] = Vp[1,:], Vp[-2,:]
        for k in range(self.N[1]-1):
            Vp_cont = Vp_cont + p_z[self.ii,self.jj,k]*Vp[self.ii,k]
        return Vp_cont

    def feasible_c(self,c):
        ii, jj = np.meshgrid(range(self.N[0] - 1), range(self.N[1] - 1), indexing='ij')
        clow = (self.grid[0][ii] - self.grid[0][-1]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][jj]) + 10**-4
        chigh = (self.grid[0][ii] - self.grid[0][0]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][jj]) - 10**-4
        return np.minimum(np.maximum(c, clow), chigh)

    def weights_indices(self,b_prime):
        #following argument has 10**-8 term because I was concerned about overflow.
        #be sure to subtract both the boundary point from the argument (to write
        #b as a multiple of Delta_a) and subtract 1.
        ind = np.array([np.floor((b-self.bnd[0][0])/self.Delta[0] + 10**-8)-1 for b in b_prime]).astype(int)
        lower = np.array([self.grid[0][0]+i*self.Delta[0] for i in ind])
        return 1 - (b_prime - lower)/self.Delta[0], ind

    def V(self,c,prob='KD'):
        B = np.exp(-self.rho*self.dt)*self.P(c,prob) - sp.eye(self.M)
        return sp.linalg.spsolve(-B, self.dt*self.u(c).reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1))

    def MPFI(self,c,V,M,prob):
        u, P, V = self.dt*self.u(c).reshape((self.M,)), self.P(c,prob), V.reshape((self.M,))
        for i in range(M+1):
            V = u + np.exp(-self.rho*self.dt)*(P*V)
        return V.reshape((self.N[0]-1,self.N[1]-1))

    def solve_MPFI(self,method,M,V_init,prob='KD'):
        if self.show_method==1:
            print("Starting MPFI with {0} policy updates, {1} probabilities, and {2} relaxations".format(method,prob,M))
        i, eps, V = 0, 1, V_init
        tic = time.time()
        while i < self.maxiter and eps > self.tol:
            if (i % 20 == 0) & (self.show_iter == 1):
                print("Iteration:", i, "Difference:", eps)
            c = self.polupdate(method,V,prob,1)
            V_prime = self.MPFI(c,V,M,prob)
            if np.max(np.isnan(c))==True:
                eps, i = -1, self.maxiter
            else:
                eps, i = np.max(np.abs(V-V_prime)), i + 1
            V = V_prime
        toc = time.time()
        if self.show_final==1:
            print("Iterations and difference:", (i, eps))
            print("Time taken:", toc-tic)
        return V, c, toc-tic, i

    def nonstat_solve(self,method,prob='KD'):
        if self.show_method==1:
            print("Starting non-stationary problem with {0} policy updates and {1} probabilities".format(method,prob))
        V = np.zeros((self.N[0]-1,self.N[1]-1,self.NA-1))
        c = np.zeros((self.N[0]-1,self.N[1]-1,self.NA-1))
        tic = time.time()
        c[:,:,self.NA-2] = self.polupdate('BF',0*self.V(self.c0,prob),prob,0)
        V[:,:,self.NA-2] = self.MPFI(c[:,:,self.NA-2],0*self.V(self.c0,prob),0,prob)
        toc = time.time()
        if self.show_method==1:
            print("Initialize with brute force")
            print("Time taken:", toc-tic)
        tic = time.time()
        for t in range(1,self.NA-1):
            V_imp = V[:,:,self.NA-1-t]
            if (int(self.NA-1-t) % 20 == 0) & (self.show_iter == 1):
                print("Period:",self.NA-1-t)
            c[:,:,self.NA-1-t-1] = self.polupdate(method,V_imp,prob,0)
            V[:,:,self.NA-1-t-1] = self.MPFI(c[:,:,self.NA-1-t-1],V_imp,0,prob)
        toc = time.time()
        if self.show_final==1:
            print("Time taken:", toc-tic)
        return V, c, toc-tic

    def solve_PFI(self,method='BF',prob='KD'):
        if self.show_method==1:
            print("Starting PFI with {0} policy updates and {1} probabilities".format(method,prob))
        i, eps, V = 0, 1, self.V(self.c0,prob)
        tic = time.time()
        while i < self.maxiter_PFI and eps > self.tol:
            if self.show_iter == 1:
                print("Iteration:", i, "Difference:", eps)
            c = self.polupdate(method,V,prob,stat=1)
            V_prime = self.V(c,prob)
            if np.max(np.isnan(c))==True:
                eps, i = -1, self.maxiter_PFI
            else:
                eps, i = np.max(np.abs(V-V_prime)), i + 1
            V = V_prime
        toc = time.time()
        if (np.max(np.isnan(c))==True) or (np.max(np.isnan(V))==True):
            print("Problem: nans encountered")
            return -1 + 0*V, -1 + 0*V, np.nan, self.maxiter_PFI
        else:
            if self.show_final==1:
                print("Iterations and difference:", (i, eps))
                print("Time taken:", toc-tic)
            return V, c, toc-tic, i

    def P_func(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M)],B[(B>-1)*(B<self.M)],C[(B>-1)*(B<self.M)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

"""
Continuous-time stationary problem
"""

class CT_stat_IFP(object):
    def __init__(self,rho=0.05, r=0.02, gamma=2., ybar=1, mubar=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, bnd=[[0,60],[-0.8,0.8]], N=(100,10),
    dt=10**-4, tol=10**-4, maxiter=200, maxiter_PFI=25, show_method=1,
    show_iter=1, show_final=1, kappac = 2):
        self.r, self.rho, self.gamma = r, rho, gamma
        self.ybar, self.mubar, self.sigma = ybar, mubar, sigma
        self.tol, self.maxiter, self.maxiter_PFI = tol, maxiter, maxiter_PFI
        self.N, self.M = N, (N[0]-1)*(N[1]-1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(2)]
        self.grid = [np.linspace(self.bnd[i][0]+self.Delta[i],self.bnd[i][1]-self.Delta[i],self.N[i]-1) for i in range(2)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],indexing='ij')
        self.ii, self.jj = np.meshgrid(range(self.N[0] - 1),range(self.N[1] - 1), indexing='ij')
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1] - 2)
        self.trans_keys = [(1,0),(-1,0),(0,1),(0,-1)]
        self.c0 = self.r*self.xx[0] + self.ybar*np.exp(self.xx[1])
        self.dt, self.Dt = dt, dt + 0*self.c0
        self.kappac = kappac
        self.cmax = self.kappac*self.c0
        self.V0 = self.V(self.c0)
        self.show_method, self.show_iter, self.show_final = show_method, show_iter, show_final

    def p_func(self,ind,c):
        ii,jj = ind
        p_func = {}
        x = (self.xx[0][ii,jj],self.xx[1][ii,jj])
        dt, c, sig = self.Dt[ii,jj], c[ii,jj], self.sigsig[ii,jj]
        d = [dt/self.Delta[i]**2 for i in range(2)]
        p_func[(1,0)] = d[0]*self.Delta[0]*np.maximum(self.r*x[0]+self.ybar*np.exp(x[1])-c,0)
        p_func[(-1,0)] = d[0]*self.Delta[0]*np.maximum(-(self.r*x[0]+self.ybar*np.exp(x[1])-c),0)
        p_func[(0,1)] = d[1]*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-x[1]),0))
        p_func[(0,-1)] = d[1]*(sig**2/2 + self.Delta[1]*np.maximum(-self.mubar*(-x[1]),0))
        return p_func

    def prob_test(self,kappac):
        diag = 1 - sum(self.p_func((self.ii, self.jj),kappac*self.c0).values())
        return np.min(diag) > 0

    def P(self,c):
        ii, jj = self.ii, self.jj
        row = ii*(self.N[1]-1) + jj
        diag = 1 - sum(self.p_func((ii,jj),c).values())
        P = self.P_func(row,row,diag)
        for key in self.trans_keys:
            column = row + key[0]*(self.N[1]-1) + key[1]
            P = P + self.P_func(row,column,self.p_func((ii,jj),c)[key])
        if np.min(P) < 0:
            print("Problem: probabilities outside of unit interval")
        else:
            return P

    def V(self,c):
        b = c**(1-self.gamma)/(1-self.gamma)
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        B = (sp.eye(self.M) - sp.diags(D)*self.P(c))/self.dt
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1))

    def solve_PFI(self):
        V, i, eps = self.V0, 1, 1
        tic = time.time()
        while i < self.maxiter_PFI and eps > self.tol:
            V1 = self.V(self.polupdate(V))
            if np.max(np.isnan(V1))==True:
                eps, i = -1, self.maxiter_PFI
            eps, i = np.amax(np.abs(V1-V)), i+1
            V = V1
            if self.show_iter == 1:
                print("Difference in iterations:", eps, "Iterations:", i)
        toc = time.time()
        c = self.polupdate(V)
        if (np.max(np.isnan(c))==True) or (np.max(np.isnan(V))==True):
            print("Problem: nans encountered")
            return -1 + 0*V, -1 + 0*V, np.nan, self.maxiter_PFI
        else:
            if self.show_final==1:
                print("Difference in PFI:", eps, "Iterations:", i)
                print("Time taken:", toc-tic)
            return V, c, toc-tic, i

    def MPFI(self,c,V,M):
        b = (self.Dt*c**(1-self.gamma)/(1-self.gamma)).reshape((self.M,))
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        V, P = V.reshape((self.M,)), sp.diags(D)*self.P(c)
        for i in range(M+1):
            V = b + P*V
        return V.reshape((self.N[0]-1,self.N[1]-1))

    def solve_MPFI(self,M):
        V, i, eps = self.V0, 1, 1
        tic = time.time()
        while i < self.maxiter and eps > self.tol:
            if (i % 20 == 0) & (self.show_iter == 1):
                print("Iteration:", i, "Difference:", eps)
            V1 = self.MPFI(self.polupdate(V),V,M)
            eps = np.amax(np.abs(V1-V))
            V, i = V1, i+1
        toc = time.time()
        print("Difference in iterates for MPFI", M, ":", eps, "Iterations:", i)
        if np.mean(self.P(self.polupdate(V)).sum(axis=1)) < 1:
            print("Problem: mass missing in transition matrix")
        return V, self.polupdate(V), toc-tic, i

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1))
        Vbig[1:-1,1:-1] = V
        VB1 = (Vbig[1:-1,1:-1]-Vbig[:-2,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1]-Vbig[1:-1,1:-1])/self.Delta[0]
        obj = lambda c: c**(1-self.gamma)/(1-self.gamma) + np.exp(-self.rho*self.Dt) \
        *(np.maximum(self.c0 - c, 0)*VF1 - np.maximum(-(self.c0 - c), 0)*VB1)
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum((np.exp(-self.rho*self.Dt)*VF1)**(-1/self.gamma), self.c0)
            chigh = np.maximum((np.exp(-self.rho*self.Dt)*VB1)**(-1/self.gamma), self.c0)
        clow[VF1<=0], chigh[VB1<=0] = self.cmax[VF1<=0], self.cmax[VB1<=0]
        runmax = np.concatenate((obj(self.c0).reshape(1,self.M), \
        obj(clow).reshape(1,self.M), obj(chigh).reshape(1,self.M)))
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]-1,self.N[1]-1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:] = np.minimum(C[0,:], self.c0[0,:])
        C[-1,:] = np.maximum(C[-1,:], self.c0[-1,:])
        return C

    def P_func(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M)],B[(B>-1)*(B<self.M)],C[(B>-1)*(B<self.M)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

"""
def mesh(self,m):
    return np.meshgrid(range(max(-m[0],0),self.N[0] - 1 - max(m[0],0)), \
    range(max(-m[1],0), self.N[1] - 1 - max(m[1],0)), indexing='ij')
Non-stationary continuous-time income fluctuation problem
"""

class CT_nonstat_IFP(object):
    def __init__(self, rho=0.05, r=0.03, gamma=2., ybar=1, mubar=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, bnd=[[0,50],[-0.6,0.6],[0,60]], N=(40,10,10),
    dt = 10**(-4), tol=10**-6, maxiter=200, maxiter_PFI=25, mono_tol=10**(-6),
    show_method=1, show_iter=1, show_final=1, kappac = 2):
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
        self.cmax = self.kappac*self.c0
        self.show_method, self.show_iter, self.show_final = show_method, show_iter, show_final

    def p_func(self,ind,c):
        ii,jj,kk = ind
        p_func, dum = {}, 0*self.xx[0][ii,jj,kk]
        x = (self.xx[0][ii,jj,kk],self.xx[1][ii,jj,kk],self.xx[2][ii,jj,kk])
        dt,c = self.Dt[ii,jj,kk], c[ii,jj,kk]
        sig = self.sigsig[ii,jj,kk]
        d = [dt/self.Delta[i]**2 for i in range(3)]
        p_func[(1,0,0)] = d[0]*self.Delta[0]*np.maximum(self.r*x[0]+self.ybar*np.exp(x[1])-c,0)
        p_func[(-1,0,0)] = d[0]*self.Delta[0]*np.maximum(-(self.r*x[0]+self.ybar*np.exp(x[1])-c),0)
        p_func[(0,1,0)] = d[1]*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-x[1]),0))
        p_func[(0,-1,0)] = d[1]*(sig**2/2 + self.Delta[1]*np.maximum(-self.mubar*(-x[1]),0))
        p_func[(0,0,1)] = dt/self.Delta[2]
        return p_func

    def P(self,c):
        ii, jj, kk = self.ii, self.jj, self.kk
        row = ii*((self.N[1]-1)*(self.N[2]-1)) + jj*(self.N[2]-1) + kk
        diag = 1 - sum(self.p_func((ii,jj,kk),c).values())
        P = self.P_func(row,row,diag)
        for key in self.trans_keys:
            column = row + key[0]*(self.N[1]-1)*(self.N[2]-1) + key[1]*(self.N[2]-1) + key[2]
            P = P + self.P_func(row,column,self.p_func((ii,jj,kk),c)[key])
        return P

    #in the following, both sides of the system are divided through by dt.
    def V(self,c):
        b = c**(1-self.gamma)/(1-self.gamma)
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        B = (sp.eye(self.M) - sp.diags(D)*self.P(c))/self.dt
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[0]-1,self.N[1]-1,self.N[2]-1))

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1, self.N[2]+1))
        Vbig[1:-1,1:-1,1:-1] = V
        VB1 = (Vbig[1:-1,1:-1,1:-1]-Vbig[:-2,1:-1,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1,1:-1]-Vbig[1:-1,1:-1,1:-1])/self.Delta[0]
        obj = lambda c: c**(1-self.gamma)/(1-self.gamma) + np.exp(-self.rho*self.Dt) \
        *(np.maximum(self.c0 - c, 0)*VF1 - np.maximum(-(self.c0 - c), 0)*VB1)
        with np.errstate(divide='ignore',invalid='ignore'):
            clow = np.minimum((np.exp(-self.rho*self.Dt)*VF1)**(-1/self.gamma), self.c0)
            chigh = np.maximum((np.exp(-self.rho*self.Dt)*VB1)**(-1/self.gamma), self.c0)
        clow[VF1<=0], chigh[VB1<=0] = self.cmax[VF1<=0], self.cmax[VB1<=0]
        runmax = np.concatenate((obj(self.c0).reshape(1,self.M), \
        obj(clow).reshape(1,self.M), obj(chigh).reshape(1,self.M)))
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]-1,self.N[1]-1,self.N[2]-1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:,:] = np.minimum(C[0,:,:], self.c0[0,:,:])
        C[-1,:,:] = np.maximum(C[-1,:,:], self.c0[-1,:,:])
        return C

    def solve_PFI(self):
        if self.show_method==1:
            print("Starting naive PFI")
        V, i, eps = self.V(self.c0), 0, 1
        tic = time.time()
        while i < self.maxiter_PFI and eps > self.tol:
            V1 = self.V(self.polupdate(V))
            eps = np.amax(np.abs(V1-V))
            if np.min(V1-V) < -self.mono_tol:
                print("PROBLEM: Failure of monotonicity at:", len(V[V1-V<-self.mono_tol]), "points out of ", self.M)
                print("Average magnitude of monotonicity failure:", np.mean((V1-V)[V1-V<-self.mono_tol]))
            V, i = V1, i+1
            if self.show_iter == 1:
                print("Difference in iterations:", eps, "Iterations:", i)
        toc = time.time()
        if self.show_final==1:
            print("Time taken:", toc-tic)
        if np.max(self.polupdate(V) < self.cmax) == False:
            print("PROBLEM: consumption inequality doesn't hold!")
        c = self.polupdate(V)
        return V, c, toc-tic, i

    #still need UP age transitions in following to ensure diagonals have correct weight
    def p_func_slice(self,ind,c):
        ii,jj = ind
        p_func = {}
        x = (self.xx[0][ii,jj,0],self.xx[1][ii,jj,0])
        dt, c, sig = self.Dt[ii,jj,0], c[ii,jj], self.sigsig[ii,jj,0]
        p_func[(1,0)] = dt*np.maximum(self.r*x[0]+np.exp(x[1])-c,0)/self.Delta[0]
        p_func[(-1,0)] = dt*np.maximum(-(self.r*x[0]+np.exp(x[1])-c),0)/self.Delta[0]
        p_func[(0,1)] = dt*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-x[1]),0))/self.Delta[1]**2
        p_func[(0,-1)] = dt*(sig**2/2 + self.Delta[1]*np.maximum(-self.mubar*(-x[1]),0))/self.Delta[1]**2
        p_func[(0,0,1)] = dt/self.Delta[2]
        return p_func

    def P_slice(self,c):
        ii, jj = np.meshgrid(range(self.N[0] - 1), range(self.N[1] - 1), indexing='ij')
        row = ii*(self.N[1]-1) + jj
        diag = 1 - sum(self.p_func_slice((ii,jj),c).values())
        P = self.P_func_slice(row,row,diag)
        for key in self.trans_keys_slice:
            column = row + key[0]*(self.N[1]-1) + key[1]
            P = P + self.P_func_slice(row,column,self.p_func_slice((ii,jj),c)[key])
        return P

    #following finds V_imp, a (N[0]-1) x (N[1]-1) array corresponding to the
    #value function at a particular age, given V_imp_up for the value function
    #at a higher age and a (N[0]-1) x (N[1]-1) policy c_slice.
    def V_imp(self,V_imp_up,c_slice):
        Dt_slice = self.Dt[:,:,0]
        b = Dt_slice*c_slice**(1-self.gamma)/(1-self.gamma) + np.exp(-self.rho*Dt_slice)*(Dt_slice/self.Delta[2])*V_imp_up
        D = np.exp(-self.rho*Dt_slice).reshape((self.M_slice,))
        B = sp.eye(self.M_slice) - sp.diags(D)*self.P_slice(c_slice)
        b, B = b/self.dt, B/self.dt
        return sp.linalg.spsolve(B, b.reshape((self.M_slice,))).reshape((self.N[0]-1,self.N[1]-1))

    def polupdate_slice(self,V_imp):
        Vbig = -10**6*np.ones((self.N[0]+1, self.N[1]+1))
        Vbig[1:-1,1:-1] = V_imp
        VB1 = (Vbig[1:-1,1:-1]-Vbig[:-2,1:-1])/self.Delta[0]
        VF1 = (Vbig[2:,1:-1]-Vbig[1:-1,1:-1])/self.Delta[0]
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
        return C

    def solveVslice(self,V_imp_up,c_guess):
        V = self.V_imp(V_imp_up,c_guess)
        eps, i = 1, 1
        while i < 20 and eps > self.tol:
            V1 = self.V_imp(V_imp_up,self.polupdate_slice(V))
            eps = np.amax(np.abs(V - V1))
            V, i = V1, i+1
        return V

    def solve_seq_imp(self):
        if self.show_method==1:
            print("Starting sequential PFI")
        V = 0*np.ones((self.N[0]-1,self.N[1]-1,self.N[2]-1))
        c = 0*np.ones((self.N[0]-1,self.N[1]-1,self.N[2]-1))
        tic = time.time()
        V_guess, c_guess = np.zeros((self.N[0]-1,self.N[1]-1)), self.c0[:,:,0]
        V[:,:,self.N[2]-2] = self.solveVslice(V_guess, c_guess)
        c[:,:,self.N[2]-2] = self.polupdate_slice(V[:,:,-1])
        for k in range(self.N[2]-2):
            if (int(self.N[2]-2-k-1) % 20 == 0) & (self.show_iter == 1):
                print("Age:", self.N[2]-2-k-1)
            V[:,:,self.N[2]-2-k-1] = self.solveVslice(V[:,:,self.N[2]-2-k],c[:,:,self.N[2]-2-k])
            c[:,:,self.N[2]-2-k-1] = self.polupdate_slice(V[:,:,self.N[2]-2-k])
        toc = time.time()
        if self.show_final==1:
            print("Time taken:", toc-tic)
        if np.max(self.polupdate(V) < self.cmax) == False:
            print("PROBLEM: consumption inequality doesn't hold!")
        c = self.polupdate(V)
        return V, c, toc-tic

    def P_func_slice(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M_slice)],B[(B>-1)*(B<self.M_slice)],C[(B>-1)*(B<self.M_slice)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M_slice,self.M_slice))

    def P_func(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M)],B[(B>-1)*(B<self.M)],C[(B>-1)*(B<self.M)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))
