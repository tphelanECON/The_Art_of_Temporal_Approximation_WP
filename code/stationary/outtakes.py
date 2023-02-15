"""
Want to understand the discrepancy in speed. It is in polupdate.
I suspect that this is partly due to erroneous interpolation. NO. it seems like the Vp_cont is much slower.

func1_time = %timeit -o -n 2000 X.weights_indices(b_prime)
#time_decomp['DT']['Weights'] = np.mean(func1_time.timings)
func1_time = %timeit -o -n 2000 X.Vp_cont(V0,prob=prob)
#time_decomp['DT']['Compute Vp'] = np.mean(func1_time.timings)
func1_time = %timeit -o -n 2000 custom_interp()
#time_decomp['DT']['Interpolate Vp'] = np.mean(func1_time.timings)
timeit(X.Vp_cont(V0,prob='KD'))
timeit(X.Vp_cont_KD(V0))
timeit(weights_indices(b_prime))

Vp_cont, Vp = np.zeros((X.N[0]-1,X.N[1]-1)), np.zeros((X.N[0]-1,X.N[1]-1))
Vp[1:-1,:] = (V0[2:,:] - V0[:-2,:])/(2*X.Delta[0])
Vp[0,:], Vp[-1,:] = Vp[1,:], Vp[-2,:]

def Vp_cont(V,prob):
    Vp_cont, Vp = np.zeros((X.N[0]-1,X.N[1]-1)), np.zeros((X.N[0]-1,X.N[1]-1))
    Vp[1:-1,:] = (V[2:,:] - V[:-2,:])/(2*X.Delta[0])
    Vp[0,:], Vp[-1,:] = Vp[1,:], Vp[-2,:]
    for k in range(X.N[1]-1):
        Vp_cont = Vp_cont + X.p_z[prob][X.ii,X.jj,k]*Vp[X.ii,k]
    return Vp_cont

timeit(weights_indices(b_prime))
timeit(Vp_cont(V0,'KD'))
np.sum([X.p_z[prob][X.ii,X.jj,k]*Vp[X.ii,k] for k in range(X.N[1]-1)])
timeit(Y.polupdate(V=V0))

"""

"""
cnew = np.zeros((X.N[0]-1,X.N[1]-1))
cand_b = np.zeros((X.N[0]-1,X.N[1]-1))
cand_c = np.zeros((X.N[0]-1,X.N[1]-1))
#candidate asset values and consumption
cand_b[X.ii,X.jj] = X.dt*X.u_prime_inv((1+X.dt*X.r)*np.exp(-X.rho*X.dt)*Vp[X.ii,X.jj]) \
+ X.grid[0][X.ii]/(1+X.dt*X.r) - X.dt*X.ybar*np.exp(X.grid[1][X.jj])
cand_c[X.ii,X.jj] = (cand_b[X.ii,X.jj] - X.grid[0][X.ii]/(1+X.dt*X.r))/X.dt + X.ybar*np.exp(X.grid[1][X.jj])
#time_decomp['DT cont'] = {}
#time_decomp['CT cont'] = {}

#time_decomp['DT cont'][l] = time_decomp['DT'][l]*time_decomp['DT']['Iterations']/time_decomp['DT']['Total']
#time_decomp['DT cont']['Total'] = 1
#time_decomp['CT cont'][l] = time_decomp['CT'][l]*time_decomp['CT']['Iterations']/time_decomp['CT']['Total']
#time_decomp['CT cont']['Total'] = 1
"""

"""
for N_t in N_t_list:
    for framework in ['DT']:
        fig, ax = plt.subplots()
        for j in range(N[1]-1):
            color = colorFader(c1,c2,j/(X[N_t].N[1]-1))
            if j in [0,X[N_t].N[1]-2]:
                inc = X[N_t].ybar*np.round(np.exp(X[N_t].grid[1][j]),2)
                #ax.plot(X[N_t].grid[0], sol[framework][N_t][0][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
                ax.plot(X[N_t].grid[0], X[N_t].V0[:,j], color=color, label="Income {0}".format(inc), linewidth=1)
            else:
                #ax.plot(X[N_t].grid[0], sol[framework][N_t][0][:,j], color=color, linewidth=1)
                ax.plot(X[N_t].grid[0], X[N_t].V0[:,j], color=color, linewidth=1)
        plt.xlabel('Assets $b$')
        plt.legend()
        plt.title('Value functions $N_t$ = {0}'.format(N_t))
        #destin = '../../main/figures/DT_V_{0}_{1}.eps'.format(pol_method, val_method)
        #plt.savefig(destin, format='eps', dpi=1000)
        plt.show()

        fig, ax = plt.subplots()
        for j in range(X[N_t].N[1]-1):
            color = colorFader(c1,c2,j/(N[1]-1))
            if j in [0,X[N_t].N[1]-2]:
                inc = X[N_t].ybar*np.round(np.exp(X[N_t].grid[1][j]),2)
                ax.plot(X[N_t].grid[0], sol[framework][N_t][1][:,j], color=color, label="Income {0}".format(inc), linewidth=1)
            else:
                ax.plot(X[N_t].grid[0], sol[framework][N_t][1][:,j], color=color, linewidth=1)
        plt.xlabel('Assets $b$')
        plt.legend()
        plt.title('Policy functions $N_t$ = {0}'.format(N_t))
        #destin = '../../main/figures/DT_c_{0}_{1}.eps'.format(pol_method, val_method)
        #plt.savefig(destin, format='eps', dpi=1000)
        plt.show()
        plt.close()

P_z, PzT, p = {}, {}, {}
P_diff = {}
for N_t in N_t_list:
    P_diff[N_t] = X[N_t].KD_z() - X[N_t].Tauchen()[0,:,:]

K = 100
for N_t in N_t_list:
    P_z[N_t] = X[N_t].KD_z()
    P_z[N_t] = P_z[N_t].round(decimals=4)
    p[N_t] = np.zeros((P_z[N_t].shape[0],K))
    p[N_t][:,-1] = 1/P_z[N_t].shape[0]
    PzT[N_t] = np.mat(P_z[N_t]).T
    for k in range(1,K):
        p[N_t][:,-1-k] = (PzT[N_t]*p[N_t][:,-k].reshape((P_z[N_t].shape[0],1))).reshape((P_z[N_t].shape[0],))

    fig, ax = plt.subplots()
    for k in range(K):
        color = colorFader(c1,c2,k/K)
        ax.plot(X[2].grid[1], p[N_t][:,k], color=color, linewidth=1)
    plt.xlabel('Log income $z$')
    plt.title('Probabilities for $N_t$ = {0}'.format(N_t))
    plt.show()

fig, ax = plt.subplots()
for N_t in N_t_list:
    #color = colorFader(c1,c2,k/K)
    ax.plot(X[2].grid[1], p[N_t][:,0], label='$N_t$ = {0}'.format(N_t), linewidth=1)
plt.xlabel('Log income $z$')
plt.legend()
plt.title('DDD')
plt.show()

for N_t in N_t_list:
    print(p[N_t][:,0])

#self.p_z[disc+'_sp'] = self.p_z[disc]
#self.p_z[disc+'_sp'][self.p_z[disc]<10**-2] = 0
#self.probs[disc+'_sp'] = self.p_func_diff((self.ii, self.jj), prob=disc+'_sp')
"""
