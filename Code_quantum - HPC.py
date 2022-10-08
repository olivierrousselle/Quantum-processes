# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import *
import matplotlib.pyplot as plt
import scipy.constants as cst
import math
import cmath
import numba as nb
from numba import cuda
import time

hbar = cst.h/(2*np.pi)
g = 9.81 
m_u = 1.660E-27
m = 1.00784*m_u # mass of antihydrogen, in kg
tg = ((2*hbar)/(m*g**2))**(1/3) # typical scales
pg = (2*hbar*m**2*g)**(1/3)
vg = pg/m
lg = hbar/pg # length scale
f = 20e3 # frequency of the trap, in kHz
zeta = np.sqrt(hbar/(2*m*2*np.pi*f)) # initial position dispersion
Delta_p = hbar/(2*zeta) # initial momentum dispersion
Delta_v = Delta_p/m
v0 = 0.8
q0 = m*v0
h = 10e-6 # height between the initial trap and the mirror, in m.
d = 5e-2 # length of the mirror, in m.
H = 0.3
tau = np.sqrt(2*H/g)
Delta_x = Delta_v*tau
Lambda = -ai_zeros(1e6)[0] # zeros of the Airy function

GQS = 1000 # number of gravitational quantum states
discretization = 100
M_X = np.linspace((d+v0*tau)-3*Delta_x, (d+v0*tau)+3*Delta_x, discretization)
M_T = np.linspace((d/v0+tau)*0.95, (d/v0+tau)*1.05, discretization)


"""Computation of the Fourier transform of \chi_n"""
freq = np.linspace(-25*pg, 25*pg, 20001)
d_freq = (max(freq)-min(freq))/(len(freq)-1)
Z = np.fft.fftshift(np.fft.fftfreq(freq.size, d=d_freq))*hbar*(2*np.pi)
dz = (max(Z)-min(Z))/(len(Z)-1)
def Chi_FT(n):
    Chi = airy(Z/lg-Lambda[n])[0]/(np.sqrt(lg)*airy(-Lambda[n])[1])
    Chi[Z<=0] = 0
    a = np.fft.ifftshift(Chi)
    A = np.fft.fft(a)
    return dz*np.fft.fftshift(A)
start_time = time.time()
Chi_FT = np.array([Chi_FT(n)/np.sqrt(2*np.pi*hbar) for n in range(1000)])
print("Time execution Fourier transform: %s seconds ---" % round(time.time() - start_time))    
#np.save('Chi-FT_25pg_60000_n=1000.npy', A)
#Chi_FT = np.load('Chi-FT_25pg_60000_n=1000.npy')/np.sqrt(2*np.pi*hbar)
M_pz = np.linspace(-25*pg, 25*pg, len(Chi_FT[0]))

def c0(n):
    return (8*np.pi*zeta**2)**(1/4)/(np.sqrt(lg)*airy(-Lambda[n])[1])*airy(h/lg-Lambda[n]+(zeta/lg)**4)[0]*np.exp((zeta/lg)**2*(h/lg-Lambda[n]+2/3*(zeta/lg)**4)) 
c = np.array([c0(n) for n in range(GQS)])

""" numba method """

@nb.jit(parallel=True)
def Pi_numba(px, py, pz, t):
    """Parallelized calculation of the velocity distribution at the end of the mirror"""
    index_pz = np.argmin(np.abs(M_pz-pz))  
    A = 0
    for n in range(GQS):
        #value = Chi_FT[n][index_pz]
        value = (Chi_FT[n][index_pz+1]-Chi_FT[n][index_pz-1])/(M_pz[index_pz+1]-M_pz[index_pz-1])*(pz-M_pz[index_pz])+Chi_FT[n][index_pz]
        A += c[n]*value*np.exp(-1j*Lambda[n]*t/tg)
    return 1/(2*np.pi*Delta_p**2)*np.exp(-((px-q0)**2+py**2)/(2*Delta_p**2))*np.abs(A)**2

def J(X, Y, Z, T, g):
    """ Computation of the annihilation current """
    t = d/np.sqrt(X**2+Y**2)*T
    t_classical = T - t      
    if t_classical<=0:
        return 0
    theta = np.arctan2(Y, X)
    x = d*np.cos(theta)
    y = d*np.sin(theta)    
    vx = (X-x)/t_classical
    vy = (Y-y)/t_classical
    vz = Z/t_classical + g*t_classical/2
    V_perp = vz - g*t_classical 
    return (m/t_classical)**3*np.abs(V_perp)*Pi_numba(m*vx, m*vy, m*vz, t)


start_time = time.time()

img_numba = np.array([[J(X, 0, -H, T, g) for X in M_X] for T in M_T])
plt.figure()
plt.pcolormesh(M_X*1000, M_T*1000, img_numba)
plt.gca().invert_yaxis() 
plt.ylabel("$T$ (ms)")
plt.xlabel("$X$ (mm)")  
clb = plt.colorbar(format='%.0e')
clb.ax.set_title('$J~(m^{-2}s^{-1})$')
plt.show()

print("Time execution Plot J numba: %s seconds ---" % round(time.time() - start_time)) 


""" CUDA method """

@cuda.jit
def Pi_cuda(px, py, pz, t):
    """Parallelized calculation of the velocity distribution at the end of the mirror"""
    pz = 0
    index_pz = int(len(M_pz)/2)
    A = 0
    for n in range(10):
        A += c[n]*Chi_FT[n][index_pz]*cmath.exp(-1j*Lambda[n]*t/tg)
    return 1/(2*math.pi*Delta_p**2)*math.exp(-((px-q0)**2+py**2)/(2*Delta_p**2))*abs(A)**2

@cuda.jit
def J_cuda(X, Y, Z, T, g):
    """ Computation of the annihilation current """
    t = d/math.sqrt(X**2+Y**2)*T
    t_classical = T - t
    x = d
    y = 0
    vx = (X-x)/t_classical
    vy = (Y-y)/t_classical
    vz = Z/t_classical + g*t_classical/2
    V_perp = vz - g*t_classical
    return (m/t_classical)**3*abs(V_perp)*Pi_cuda(m*vx, m*vy, m*vz, t)

@cuda.jit
def plot_cuda(img_cuda):
    i, j = cuda.grid(2)
    #i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    #j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if i < img_cuda.shape[0] and j < img_cuda.shape[1]:
        X = M_X[i]
        T = M_T[j]
        img_cuda[i][j] = J_cuda(X, 0, -H, T, g)

start_time = time.time()

img_cuda = np.zeros((M_X.shape[0], M_T.shape[0]))
threadsperblock = (16, 16)
blockspergrid = (math.ceil(img_cuda.shape[0] / threadsperblock[0]), math.ceil(img_cuda.shape[1] / threadsperblock[1]))
plot_cuda[threadsperblock, blockspergrid](img_cuda)
np.sum(img_cuda)

plt.figure()
plt.pcolormesh(M_X*1000, M_T*1000, img_cuda) 
plt.gca().invert_yaxis()
plt.ylabel("$T$ (ms)")
plt.xlabel("$X$ (mm)")  
clb = plt.colorbar(format='%.0e')
clb.ax.set_title('$J~(m^{-2}s^{-1})$')
plt.show()

print("Time execution Plot J cuda: %s seconds ---" % round(time.time() - start_time)) 