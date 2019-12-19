# coding: utf-8
import numpy as np
from time import time
import sys

def find_primes(maxN):
    x1 = np.arange(maxN + 1, dtype=np.int64)
    b1 = np.zeros(np.shape(x1), dtype=np.bool)
    b1[x1 > 1] = True
    maxN2 = np.int64(maxN**(0.5) + 1)
    for n in range(2, maxN2 + 1):
        b1[2*n::n] = False
    return x1[b1]


def prime_factors(N):
    pNums = find_primes(N//2 + 1)
    pExps = np.zeros(np.shape(pNums), dtype=int)
    max_exp = int(np.log(N)/np.log(2))
    for n in range(1, max_exp + 1):
        pExps[np.mod(N, pNums**n) == 0] = n
    pN = pNums[pExps > 0]
    pE = pExps[pExps > 0]
    if 0 < np.size(pN) < 10:
        disp_pf(N, pN, pE)
    elif np.size(pN) == 0:
        print('{N} is a prime number!'.format(N=N))
    else:
        pass
    return pN, pE


def find_lcm(num_array):
    Nmax = max(num_array)
    pNums = find_primes(Nmax + 1)
    pExps = np.zeros(np.shape(pNums), dtype=int)
    for N in num_array:
        pExps2 = np.zeros(np.shape(pNums), dtype=int)
        if N in pNums:
            pExps2[pNums == N] = 1
        else:
            max_exp = int(np.log(N)/np.log(2))
            for n in range(1, max_exp + 1):
                pExps2[np.mod(N, pNums**n) == 0] = n
        pExps = np.maximum(pExps, pExps2)
    pN = pNums[pExps > 0]
    pE = pExps[pExps > 0]
    outN = np.product(pN**pE)
    if 0 < np.size(pN) < 10:
        disp_pf(outN, pN, pE)
    else:
        pass
    return outN, pN, pE
    

def disp_pf(N, pNums, pExps):
    factors1 = []
    for n, e in zip(pNums, pExps):
        if e > 1:
            factor = '{n:,d}^{e}'.format(n=n, e=e)
        else:
            factor = '{n:,d}'.format(n=n)
        factors1.append(factor)
    print('\n{N:,d} = '.format(N=N) + ' * '.join(factors1))


def test_fun1(upper_limit):
    t0 = time()
    prime_array = find_primes(upper_limit)
    t1 = time()
    Nprime = np.size(prime_array)
    print('\nFound {0:,d} prime numbers in {1:.4e} sec'.format(Nprime, t1 - t0))
    print('\nOr, ~{0:,d} prime numbers per second'.format(int(Nprime/(t1 - t0))))
    print('\nPython version: {0}'.format(sys.version))
    return prime_array


if __name__ == "__main__":
    out1 = test_fun1(np.int64(1e+8))
#    num_array = list(range(2, 11))
#    lcm, pN, pE = find_lcm(num_array)
    
