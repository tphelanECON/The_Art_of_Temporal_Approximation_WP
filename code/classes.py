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
    * One needs sufficient dissaving at the top of asset grid.
    * In the DT_IFP and CT_IFP cases there is no need for an arbitrary ind
    argument, because these only ever take ii,jj = self.ii, self.jj. However, this
    generality is necessary in the nonstationary problem, where "slices" arise.
    * The DT_IFP class constructor allows for both the same probability
    transitions in the continuous-time case, denoted prob='KD' for "Kushner and
    Dupuis", and also the method of Tauchen (1986), denoted prob='Tauchen'.
    * All class constructors write self.cmax = self.kappac*self.c0, a vector
    that is a multiple of zero net saving. This default is 2 for the stationary
    problem, which never binds at the optimum.
    * N+1 points in each dimension inclusive of bnd. Ensures Delta = (bnd[1]-bnd[0])/N.

Discussion of boundaries:
    * In CT, V vanishes at Abar. We only compute values at Abar - DeltaA, Abar - 2*DeltaA, etc.
    * This is also true in the DT case (see nonstat_solve).

I discovered that the biggest time sink is in the interpolation in the brute force case.
"""

import numpy as np
from numba import jit
import scipy.sparse as sp
from scipy.stats import norm
from scipy.sparse import linalg
from scipy.sparse import diags
from scipy.interpolate import interp1d
import itertools, time

class DT_IFP(object):
    def __init__(self, rho=0.05, r=0.02, gamma=2, ybar=1, mubar=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, bnd=[[0,60],[-0.8,0.8]], N=[100,10],
    NA=60, N_t=10, N_c=4000, dt=1, tol=10**-6, maxiter=400, maxiter_PFI=25,
    show_method=1, show_iter=1, show_final=1, kappac=2):
        self.rho, self.r, self.gamma = rho, r, gamma
        self.ybar, self.mubar, self.sigma = ybar, mubar, sigma
        self.N, self.bnd, self.NA, self.N_t = N, bnd, NA, N_t
        self.tol, self.maxiter, self.maxiter_PFI = tol, maxiter, maxiter_PFI
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(2)]
        self.grid = [np.linspace(self.bnd[i][0],self.bnd[i][1],self.N[i]+1) for i in range(2)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],indexing='ij')
        self.ii, self.jj = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),indexing='ij')
        #set volatility to zero on boundary:
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1])
        self.dt, self.Dt = dt, dt + 0*self.xx[0]
        self.grid_A = np.linspace(0,self.dt*self.NA,self.NA+1)
        self.N_c, self.M = N_c, (self.N[0]+1)*(self.N[1]+1)
        #following is "zero net saving" and a natural guess for policy function:
        self.c0 = self.r*self.xx[0]/(1+self.dt*self.r) + self.ybar*np.exp(self.xx[1])
        self.dt_small = self.dt/self.N_t
        self.kappac = kappac
        self.cmax = self.kappac*self.c0
        self.show_method, self.show_iter, self.show_final = show_method, show_iter, show_final
        self.iter_keys = list(itertools.product(range(self.N[0]+1), range(self.N[1]+1)))
        self.probs, self.p_z = {}, {}
        self.p_z['KD'] = self.KD()
        self.p_z['Tauchen'] = self.Tauchen()
        self.check = np.min(self.p_z['KD']) > -10**-8
        #compute following transitions to avoid repeatedly building z matrix:
        for disc in ['KD','Tauchen']:
            self.probs[disc] = self.p_func((self.ii, self.jj), prob=disc)
        #upper and lower bounds on consumption:
        #only need to impose state constraints at boundary for CT. Not the case here
        #though as the state can move "non-locally".
        self.clow = (self.grid[0][self.ii] - self.grid[0][-1]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj]) + 10**-4
        self.chigh = (self.grid[0][self.ii] - self.grid[0][0]/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(self.grid[1][self.jj]) - 10**-4
        #following is initial guess:
        self.V0 = self.V(self.c0)
        #call following at arbitrary cont. value s.t. initialization not included in run times
        self.first_Vp = self.Vp_cont_jit(self.V0,'KD')

    def u(self,c):
        if self.gamma==1:
            return np.log(c)
        else:
            return c**(1-self.gamma)/(1-self.gamma)

    def u_prime_inv(self,x):
        return x**(-1/self.gamma)

    #dictionary of transition probabilities with range of integers as keys.
    #takes indices (giving grid location) and discretization method as arguments
    def p_func(self,ind,prob='KD'):
        ii,jj = ind
        p_func = {}
        P = self.p_z[prob]
        for k in range(self.N[1]+1):
            p_func[k] = P[ii,jj,k]
        return p_func

    #Tauchen transition matrix:
    def Tauchen(self):
        p = np.zeros((self.N[0]+1,self.N[1]+1,self.N[1]+1))
        #transition to interior z:
        iii, jjj, kkk = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),range(1,self.N[1]),indexing='ij')
        up_bnd = (self.grid[1][kkk] + self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(np.sqrt(self.dt)*self.sigma)
        down_bnd = (self.grid[1][kkk] - self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(np.sqrt(self.dt)*self.sigma)
        p[iii,jjj,kkk] = norm.cdf(up_bnd) - norm.cdf(down_bnd)
        #transition to lower bound of z:
        iii, jjj, kkk = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),0,indexing='ij')
        up_bnd = (self.grid[1][kkk] + self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(np.sqrt(self.dt)*self.sigma)
        p[iii,jjj,kkk] = norm.cdf(up_bnd)
        #transition to upper bound of z:
        iii, jjj, kkk = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),self.N[1],indexing='ij')
        down_bnd = (self.grid[1][kkk] - self.Delta[1]/2 - (1 - self.dt*self.mubar)*self.grid[1][jjj])/(np.sqrt(self.dt)*self.sigma)
        p[iii,jjj,kkk] = 1 - norm.cdf(down_bnd)
        return p

    #(N_1 - 1) x (N_1 - 1) matrix of income transitions.
    #make a KD matrix for small timestep and then iterate on transpose.
    def KD_z(self):
        p_z_small = np.zeros((self.N[1]+1,self.N[1]+1))
        #define volatility on one-dimensional grid and set to zero on boundary.
        sig = self.sigma + 0*self.grid[1]
        sig[0], sig[-1] = 0, 0
        pup_z_small = self.dt_small*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-self.grid[1]),0))/self.Delta[1]**2
        pdown_z_small = self.dt_small*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(self.grid[1]),0))/self.Delta[1]**2
        p_z_small[range(self.N[1]+1),range(self.N[1]+1)] = 1 - pup_z_small - pdown_z_small
        p_z_small[range(self.N[1]),range(1,self.N[1]+1)] = pup_z_small[:-1]
        p_z_small[range(1,self.N[1]+1),range(self.N[1])] = pdown_z_small[1:]
        p_z_T = np.linalg.matrix_power(np.mat(p_z_small).T,self.N_t)
        p_z = np.array(p_z_T.T)
        return p_z

    #Kushner and Dupuis transition matrix (main one used in paper):
    def KD(self):
        p = np.zeros((self.N[0]+1,self.N[1]+1,self.N[1]+1))
        p_z = self.KD_z()
        iii, jjj, kkk = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),range(self.N[1]+1),indexing='ij')
        p[iii,jjj,kkk] = p_z[jjj,kkk]
        return p

    #transition matrix over (b,z). Remember reshape goes l-to-r then down.
    #calls probs for z so that it doesn't have to be repeatedly calculated.
    def P(self,c,prob='KD'):
        P = sp.coo_matrix((self.M,self.M))
        b_prime = (1 + self.dt*self.r)*(self.xx[0] + self.dt*(self.ybar*np.exp(self.xx[1]) - c))
        #get location and probability weights for future assets:
        alpha, ind = self.weights_indices(b_prime)
        ii, jj = self.ii, self.jj
        for key in range(self.N[1]+1):
            #lower then upper weights on asset transition:
            row, col = ii*(self.N[1]+1) + jj, ind[ii,jj]*(self.N[1]+1) + key
            P = P + self.P_func(row,col,alpha[ii,jj]*self.probs[prob][key])
            P = P + self.P_func(row,col+(self.N[1]+1),(1-alpha[ii,jj])*self.probs[prob][key])
        return P

    #stat indicates stationary or nonstationary, which affects consumption bounds.
    #somewhat ad-hoc: set upper bound 10 x higher in nonstatioary case.
    #method=='BF' for brute force or EGM:
    def polupdate(self,method,V,prob,stat=1):
        cnew = np.zeros((self.N[0]+1,self.N[1]+1))
        if method=='BF':
            #expected value of future utility, as function of consumption guess:
            ii_, jj_, cc_ = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),range(self.N_c),indexing='ij')
            if stat==1:
                cgrid_rest = np.linspace(np.maximum(self.clow,10**-8), np.minimum(self.chigh, np.max(self.cmax)), self.N_c)
            else:
                cgrid_rest = np.linspace(np.maximum(self.clow,10**-8), np.minimum(self.chigh, 10*np.max(self.cmax)), self.N_c)
            #cgrid_rest is 3D. Need to transpose s.t. c is in the last dimension.
            cgrid_rest = np.transpose(cgrid_rest, axes=[1, 2, 0])
            for i in range(self.N[0]+1):
                for j in range(self.N[1]+1):
                    V_cont = np.zeros((self.N_c,))
                    b_prime = (1 + self.dt*self.r)*(self.grid[0][i] + self.dt*(self.ybar*np.exp(self.grid[1][j]) - cgrid_rest[i,j,:]))
                    #V_interp = [interp1d(self.grid[0], V[:,k])(b_prime) for k in range(self.N[1]+1)]
                    V_interp = [np.interp(b_prime, self.grid[0], V[:,k]) for k in range(self.N[1]+1)]
                    for k in range(self.N[1]+1):
                        V_cont = V_cont + self.p_z[prob][i,j,k]*V_interp[k]
                    RHS = self.dt*self.u(cgrid_rest[i,j,:]) + np.exp(-self.rho*self.dt)*V_cont
                    cnew[i,j] = cgrid_rest[i,j,np.argmin(-RHS)]
        else:
            #now follow EGM:
            bb, zz = self.grid[0][self.ii], self.grid[1][self.jj]
            cand_b = self.dt*self.u_prime_inv((1+self.dt*self.r)*np.exp(-self.rho*self.dt) \
            *self.Vp_cont_jit(V,prob)) + bb/(1+self.dt*self.r) - self.dt*self.ybar*np.exp(zz)
            cand_c = (cand_b - bb/(1+self.dt*self.r))/self.dt + self.ybar*np.exp(zz)
            for j in range(self.N[1]+1):
                cnew[:,j] = interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(self.grid[0])
            cnew = np.minimum(np.maximum(cnew, self.clow), self.chigh)
        return cnew

    #rapid construction of continuation values Vp (cannot call self. attributes)
    @staticmethod
    @jit(nopython=True)
    def cont_jit(future,p,N):
        cont = np.zeros((N[0]+1,N[1]+1))
        #k loops over future income, i over assets, and j over current income
        for k in range(N[1]+1):
            for i in range(N[0]+1):
                for j in range(N[1]+1):
                    cont[i,j] += p[i,j,k]*future[i,k]
        return cont

    def Vp_cont_jit(self,V,prob):
        Vp = np.zeros((self.N[0]+1,self.N[1]+1))
        Vp[1:-1,:] = (V[2:,:] - V[:-2,:])/(2*self.Delta[0])
        Vp[0,:] = (V[1,:] - V[0,:])/self.Delta[0]
        Vp[-1,:] = (V[-1,:] - V[-2,:])/self.Delta[0]
        return self.cont_jit(Vp,self.p_z[prob],self.N)

    #alpha in following is weight on LOWER asset point. I.e. if blow = self.grid[0][0]+ind*self.Delta[0]
    #then b_prime = alpha*blow + (1-alpha)*(blow+self.Delta[0]) so
    #b_prime = blow + (1-alpha)*self.Delta[0], (b_prime - blow)/self.Delta[0] = 1-alpha
    def weights_indices(self,b_prime):
        #10**-8 term in following due to rounding concern. subtract boundary
        #point from argument (to write b as multiple of Delta_a).
        ind = np.array([np.floor((b-self.bnd[0][0])/self.Delta[0] + 10**-8) for b in b_prime]).astype(int)
        return 1 - (b_prime - (self.grid[0][0]+ind*self.Delta[0]))/self.Delta[0], ind

    def V(self,c,prob='KD'):
        B = np.exp(-self.rho*self.dt)*self.P(c,prob) - sp.eye(self.M)
        return sp.linalg.spsolve(-B, self.dt*self.u(c).reshape((self.M,))).reshape((self.N[0]+1,self.N[1]+1))

    def MPFI(self,c,V,M,prob):
        flow, P, V = self.dt*self.u(c).reshape((self.M,)), self.P(c,prob), V.reshape((self.M,))
        for i in range(M+1):
            V = flow + np.exp(-self.rho*self.dt)*(P*V)
        return V.reshape((self.N[0]+1,self.N[1]+1))

    #often faster to NOT build matrix and instead write out relevant algebra
    def Jacobi_no_matrix(self,c,V,prob):
        PV = np.zeros((self.N[0]+1,self.N[1]+1))
        b_prime = (1 + self.dt*self.r)*(self.xx[0] + self.dt*(self.ybar*np.exp(self.xx[1]) - c))
        alpha, ind = self.weights_indices(b_prime)
        ii, jj = self.ii, self.jj
        #iterate over log income shocks
        for key in range(self.N[1]+1):
            #lower then upper weights on asset transition:
            row, col = ii*(self.N[1]+1) + jj, ind[ii,jj]*(self.N[1]+1) + jj + key
            ind_bnd = (col>-1)*(col<self.M)
            ii_, jj_ = ii[ind_bnd], jj[ind_bnd]
            ind_ = ind[ii_,jj_]
            alpha_, prob_ = alpha[ii_,jj_], self.probs[prob][key][ii_,jj_]
            PV[ii_,jj_] = PV[ii_,jj_] + alpha_*prob_*V[ind_, key] + (1-alpha_)*prob_*V[ind_+1, key]
        return self.dt*self.u(c) + np.exp(-self.rho*self.dt)*PV

    def solve_MPFI(self,method,M,V_init,prob='KD'):
        if self.show_method==1:
            print("Starting MPFI with {0} policy updates, {1} probabilities, and {2} relaxations".format(method,prob,M))
        i, eps, V = 0, 1, V_init
        tic = time.time()
        while i < self.maxiter and eps > self.tol:
            if (i % 10 == 0) & (self.show_iter == 1):
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

    def nonstat_solve(self,method,prob='KD',matrix=True):
        if self.show_method==1:
            print("Starting non-stationary problem with {0} policy updates and {1} probabilities".format(method,prob))
        V = np.zeros((self.N[0]+1,self.N[1]+1,self.NA+1))
        c = np.zeros((self.N[0]+1,self.N[1]+1,self.NA+1))
        time_array = np.zeros((2,self.NA+1))
        tic = time.time()
        #Timing. Death occurs AT age NA. They consume at penultimate date.
        #NA+1 gridpoints so index=NA is death and NA-1 last point for consumption.
        c[:,:,self.NA-1] = self.ybar*np.exp(self.xx[1]) + self.xx[0]/self.dt
        V[:,:,self.NA-1] = self.MPFI(c[:,:,self.NA-1],np.zeros((self.N[0]+1,self.N[1]+1)),0,prob)
        toc = time.time()
        if self.show_method==1:
            print("Initialize with known solution")
            print("Time taken:", toc-tic)
        tic = time.time()
        for k in range(self.NA-1):
            if (int(self.NA-1-k-1) % 10 ==0):
                print("Age:",self.NA-1-k-1)
            tc1 = time.time()
            c[:,:,self.NA-1-k-1] = self.polupdate(method,V[:,:,self.NA-1-k],prob,0)
            tc2 = time.time()
            tV1 = time.time()
            if matrix==True:
                V[:,:,self.NA-1-k-1] = self.MPFI(c[:,:,self.NA-1-k-1],V[:,:,self.NA-1-k],0,prob)
            else:
                V[:,:,self.NA-1-k-1] = self.Jacobi_no_matrix(c[:,:,self.NA-1-k-1],V[:,:,self.NA-1-k],prob)
            tV2 = time.time()
            time_array[:,self.NA-1-k-1] = tc2-tc1, tV2-tV1
        toc = time.time()
        if self.show_final==1:
            print("Time taken:", toc-tic)
        return V, c, toc-tic, time_array

    def solve_PFI(self,method='BF',prob='KD'):
        if self.show_method==1:
            print("Starting PFI with {0} policy updates and {1} probabilities".format(method,prob))
        i, eps, V = 0, 1, self.V(self.c0,prob)
        tic0 = time.time()
        while i < self.maxiter_PFI and eps > self.tol:
            if self.show_iter == 1:
                print("Iteration:", i, "Difference:", eps)
            tic=time.time()
            c = self.polupdate(method,V,prob,stat=1)
            V_prime = self.V(c,prob)
            toc=time.time()
            #print("Time for one iteration:",toc-tic)
            if np.max(np.isnan(c))==True:
                eps, i = -1, self.maxiter_PFI
            else:
                eps, i = np.max(np.abs(V-V_prime)), i + 1
            V = V_prime
        toc0 = time.time()
        if (np.max(np.isnan(c))==True) or (np.max(np.isnan(V))==True):
            print("Problem: nans encountered")
            return -1 + 0*V, -1 + 0*V, np.nan, self.maxiter_PFI
        else:
            if self.show_final==1:
                print("Iterations and difference:", (i, eps))
                print("Time taken:", toc-tic)
            return V, c, toc0-tic0, i

    #following used in the contruction of transition matrix P.
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
        self.N, self.M = N, (N[0]+1)*(N[1]+1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(2)]
        self.grid = [np.linspace(self.bnd[i][0],self.bnd[i][1],self.N[i]+1) for i in range(2)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],indexing='ij')
        self.ii, self.jj = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1), indexing='ij')
        #make sure volatility vanishes at boundaries:
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1])
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
        row = ii*(self.N[1]+1) + jj
        diag = 1 - sum(self.p_func((ii,jj),c).values())
        P = self.P_func(row,row,diag)
        for key in self.trans_keys:
            column = row + key[0]*(self.N[1]+1) + key[1]
            P = P + self.P_func(row,column,self.p_func((ii,jj),c)[key])
        if np.min(P) < 0:
            print("Problem: probabilities outside of unit interval")
        else:
            return P

    def V(self,c):
        b = c**(1-self.gamma)/(1-self.gamma)
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        B = (sp.eye(self.M) - sp.diags(D)*self.P(c))/self.dt
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[0]+1,self.N[1]+1))

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
        return V.reshape((self.N[0]+1,self.N[1]+1))

    def solve_MPFI(self,M):
        V, i, eps = self.V0, 1, 1
        tic = time.time()
        while i < self.maxiter and eps > self.tol:
            if (i % 10 == 0) & (self.show_iter == 1):
                print("Iteration:", i, "Difference:", eps)
            V1 = self.MPFI(self.polupdate(V),V,M)
            eps = np.amax(np.abs(V1-V))
            V, i = V1, i+1
        toc = time.time()
        print("Difference in iterates for MPFI", M, ":", eps, "Iterations:", i)
        return V, self.polupdate(V), toc-tic, i

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+3, self.N[1]+3))
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
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]+1,self.N[1]+1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:] = np.minimum(C[0,:], self.c0[0,:])
        C[-1,:] = np.maximum(C[-1,:], self.c0[-1,:])
        return C

    def P_func(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M)],B[(B>-1)*(B<self.M)],C[(B>-1)*(B<self.M)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))

"""
Non-stationary continuous-time income fluctuation problem
"""

class CT_nonstat_IFP(object):
    def __init__(self, rho=0.05, r=0.03, gamma=2., ybar=1, mubar=-np.log(0.95),
    sigma=np.sqrt(-2*np.log(0.95))*0.2, bnd=[[0,50],[-0.6,0.6],[0,60]], N=(40,10,10),
    dt = 10**(-4), tol=10**-6, maxiter=200, maxiter_PFI=25, mono_tol=10**(-6),
    show_method=1, show_iter=1, show_final=1, kappac=2):
        self.r, self.rho, self.gamma = r, rho, gamma
        self.ybar, self.mubar, self.sigma = ybar, mubar, sigma
        self.tol, self.mono_tol = tol, mono_tol
        self.maxiter, self.maxiter_PFI = maxiter, maxiter_PFI
        self.N, self.M, self.M_slice = N, (N[0]+1)*(N[1]+1)*(N[2]+1), (N[0]+1)*(N[1]+1)
        self.bnd, self.Delta = bnd, [(bnd[i][1]-bnd[i][0])/self.N[i] for i in range(3)]
        self.grid = [np.linspace(self.bnd[i][0],self.bnd[i][1],self.N[i]+1) for i in range(3)]
        self.xx = np.meshgrid(self.grid[0],self.grid[1],self.grid[2],indexing='ij')
        self.ii, self.jj, self.kk = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1),range(self.N[2]+1), indexing='ij')
        self.sigsig = self.sigma*(self.jj > 0)*(self.jj < self.N[1])
        self.trans_keys = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1)]
        self.trans_keys_slice = [(1,0),(-1,0),(0,1),(0,-1)]
        self.c0 = self.r*self.xx[0] + self.ybar*np.exp(self.xx[1])
        self.dt, self.Dt = dt, dt + 0*self.c0
        self.kappac = kappac
        #self.cmax = self.kappac*self.c0
        self.cmax = 1.5*self.xx[0][-1] + 0*self.c0
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
        row = ii*(self.N[1]+1)*(self.N[2]+1) + jj*(self.N[2]+1) + kk
        diag = 1 - sum(self.p_func((ii,jj,kk),c).values())
        P = self.P_func(row,row,diag)
        for key in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]:
            column = row + key[0]*(self.N[1]+1)*(self.N[2]+1) + key[1]*(self.N[2]+1) + key[2]
            P = P + self.P_func(row,column,self.p_func((ii,jj,kk),c)[key])
        #treat aging step separately. Do not want probabilities to "wrap around"
        key = (0,0,1)
        int_kk = self.kk<self.N[2]
        column = row + key[0]*(self.N[1]+1)*(self.N[2]+1) + key[1]*(self.N[2]+1) + key[2]
        P = P + self.P_func(row[int_kk],column[int_kk],self.p_func((ii,jj,kk),c)[key][int_kk])
        return P

    #in the following, both sides of the system are divided through by dt.
    def V(self,c):
        b = c**(1-self.gamma)/(1-self.gamma)
        D = np.exp(-self.rho*self.Dt).reshape((self.M,))
        B = (sp.eye(self.M) - sp.diags(D)*self.P(c))/self.dt
        return sp.linalg.spsolve(B, b.reshape((self.M,))).reshape((self.N[0]+1,self.N[1]+1,self.N[2]+1))

    def polupdate(self,V):
        Vbig = -10**6*np.ones((self.N[0]+3, self.N[1]+3, self.N[2]+3))
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
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]+1,self.N[1]+1,self.N[2]+1)
        C = (IND==0)*self.c0 + (IND==1)*clow + (IND==2)*chigh
        C[0,:,:] = np.minimum(C[0,:,:], self.c0[0,:,:])
        C[-1,:,:] = np.maximum(C[-1,:,:], self.c0[-1,:,:])
        return C

    #"naive" PFI: literally PFI applied to situation with stochastic aging.
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
        c = self.polupdate(V)
        return V, c, toc-tic, i

    #still need UP age transitions in following to ensure diagonals have correct weight
    def p_func_slice(self,ind,c):
        ii,jj = ind
        p_func = {}
        x = (self.xx[0][ii,jj,0],self.xx[1][ii,jj,0])
        dt, c, sig = self.Dt[ii,jj,0], c[ii,jj], self.sigsig[ii,jj,0]
        p_func[(1,0)] = dt*np.maximum(self.r*x[0]+self.ybar*np.exp(x[1])-c,0)/self.Delta[0]
        p_func[(-1,0)] = dt*np.maximum(-(self.r*x[0]+self.ybar*np.exp(x[1])-c),0)/self.Delta[0]
        p_func[(0,1)] = dt*(sig**2/2 + self.Delta[1]*np.maximum(self.mubar*(-x[1]),0))/self.Delta[1]**2
        p_func[(0,-1)] = dt*(sig**2/2 + self.Delta[1]*np.maximum(-self.mubar*(-x[1]),0))/self.Delta[1]**2
        p_func[(0,0,1)] = dt/self.Delta[2]
        return p_func

    def P_slice(self,c):
        ii, jj = np.meshgrid(range(self.N[0]+1), range(self.N[1]+1), indexing='ij')
        row = ii*(self.N[1]+1) + jj
        diag = 1 - sum(self.p_func_slice((ii,jj),c).values())
        P = self.P_func_slice(row,row,diag)
        for key in self.trans_keys_slice:
            column = row + key[0]*(self.N[1]+1) + key[1]
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
        return sp.linalg.spsolve(B, b.reshape((self.M_slice,))).reshape((self.N[0]+1,self.N[1]+1))

    def polupdate_slice(self,V_imp):
        Vbig = -10**6*np.ones((self.N[0]+3, self.N[1]+3))
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
        IND = np.argmax(runmax,axis=0).reshape(self.N[0]+1,self.N[1]+1)
        C = (IND==0)*c0 + (IND==1)*clow + (IND==2)*chigh
        #force dissaving at top and saving at bottom:
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
        V = np.zeros((self.N[0]+1,self.N[1]+1,self.N[2]+1))
        c = np.zeros((self.N[0]+1,self.N[1]+1,self.N[2]+1))
        time_array = np.zeros((2,self.N[2]+1))
        V_guess, c_guess = np.zeros((self.N[0]+1,self.N[1]+1)), self.c0[:,:,0]
        V[:,:,self.N[2]-1] = self.solveVslice(V_guess, c_guess)
        c[:,:,self.N[2]-1] = self.polupdate_slice(V[:,:,-1])
        tic = time.time()
        for k in range(self.N[2]-1):
            if (int(self.N[2]-1-k-1) % 10 == 0) & (self.show_iter == 1):
                print("Age:", self.N[2]-1-k-1)
            tV1 = time.time()
            V[:,:,self.N[2]-1-k-1] = self.solveVslice(V[:,:,self.N[2]-1-k],c[:,:,self.N[2]-1-k])
            tV2 = time.time()
            tc1 = time.time()
            c[:,:,self.N[2]-1-k-1] = self.polupdate_slice(V[:,:,self.N[2]-1-k-1])
            tc2 = time.time()
            time_array[:,self.N[2]-1-k-1] = tc2-tc1, tV2-tV1
        toc = time.time()
        if self.show_final==1:
            print("Time taken:", toc-tic)
        return V, c, toc-tic, time_array

    def P_func_slice(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M_slice)],B[(B>-1)*(B<self.M_slice)],C[(B>-1)*(B<self.M_slice)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M_slice,self.M_slice))

    #following used in matrix construction. Only want to pass arguments to
    #sparse matrix builder that actually lie on grid. B=columns.
    def P_func(self,A,B,C):
        A,B,C = np.array(A), np.array(B), np.array(C)
        A,B,C = A[(B>-1)*(B<self.M)],B[(B>-1)*(B<self.M)],C[(B>-1)*(B<self.M)]
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.M,self.M))
