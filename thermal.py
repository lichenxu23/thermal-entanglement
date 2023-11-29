import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
I = np.eye(2)

def single_site_operator(operator, site, N):
    op_list = [I] * N
    op_list[site] = operator
    result = scipy.sparse.identity(1)
    for op in op_list:
        result = scipy.sparse.kron(result, op)
    return result

def two_site_operator(op1, site1, op2, site2, N):
    return single_site_operator(op1, site1, N) * single_site_operator(op2, site2, N)

def tfim(N,J,h=1):
    # H = -J \sum ZZ - h \sum X
    H = np.zeros((2**N,2**N))
    
    for i in range(N):
        H -= h * single_site_operator(X,i,N)
        H -= J * two_site_operator(Z,i,Z,(i+1)%N,N)
    
    return H

def XXZ(N,lam):
    # H = \sum ZZ + lambda(XX + YY)
    H = np.zeros((2**N,2**N))

    for i in range(N):
        H += two_site_operator(Z,i,Z,(i+1)%N,N)
        H += lam * two_site_operator(Y,i,Y,(i+1)%N,N)
        H += lam * two_site_operator(X,i,X,(i+1)%N,N)

    return H

def log_negativity(rho):
    # calculates the log negativity of rho, with the bipartition being at the center
    shape = np.shape(rho)
    N = np.log2(shape[0])
    M = int(2 ** (N/2))
    rhoTA = rho.reshape(M,M,M,M).swapaxes(0, 2).reshape(shape)
    singular_values = np.linalg.svd(rhoTA, compute_uv=False)
    log_neg = np.log(np.sum(singular_values))
    
    return log_neg

if __name__ == '__main__':

    #J = 2
    #h = 1
    #lam = 1

    #for J in tqdm(np.linspace(0,5,10)):
    for lam in tqdm(np.linspace(0,2,10)):
    #for J in [2]:
    #for lam in [1]:
        plt.figure()
        plt.xlabel(r'$\beta$')
        plt.xscale('log')
        plt.ylabel(r'$N$')
        for N in tqdm([4,6,8,10]):
            #H = tfim(N,J,1)
            H = XXZ(N,lam)
            energies, eigenstates = scipy.linalg.eigh(H)
            betas = np.logspace(-2,1,num=10,base = 10)
            log_negs = []
            for beta in tqdm(betas):
                rho = sum(np.exp(-beta * en) * np.outer(st, st.conj()) for en, st in zip(energies, eigenstates))
                rho = rho/np.trace(rho)
                log_negs.append(log_negativity(rho))
            plt.scatter(betas, log_negs, label=f'N = {N}')
        plt.legend()
        #plt.savefig(f'/home/chenxu/thermal/tfim J={J}.png')
        plt.savefig(f'/home/chenxu/thermal/XXZ lambda={lam}.png')
