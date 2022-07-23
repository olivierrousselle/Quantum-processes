# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import *
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.constants as cst
import numba as nb
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
delta_E = 10 # photodetachement recoil energy, in micro-eV
ve = 0.3227739549*np.sqrt(delta_E) # photodetachment recoil velocity, in m/s
qe = m*ve
h = 10e-6 # height between the initial trap and the mirror, in m.
d = 5e-2 # length of the mirror, in m.
Lambda = -ai_zeros(1e6)[0] # zeros of the Airy function
theta_n = np.pi/2 # horizontal polarization

GQS = 1000 # number of gravitational quantum states


"""Calculation of the initial distribution in velocity"""
def I(sigma, theta):
    return (sigma**2/np.tanh(1/sigma**2)-sigma**4)*np.sin(theta)**2 + (1-2*(sigma**2/np.tanh(1/sigma**2)-sigma**4))*np.cos(theta)**2
def Pi_0(Vx, Vy, Vz):
    Vvec = np.array([Vx,Vy,Vz])
    V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    polar = np.array([0, np.sin(theta_n), np.cos(theta_n)])
    theta_v_n = np.arccos(np.vdot(Vvec, polar)/V)
    return (1/(np.sqrt(2*np.pi)*Delta_v)*(np.exp(-(V-ve)**2/(2*Delta_v**2))-np.exp(-(V+ve)**2/(2*Delta_v**2)))*3/(4*np.pi*V*ve))*I(Delta_v/np.sqrt(V*ve),theta_v_n)

"""Plot of the initial distribution in velocity"""
M_vy = np.linspace(-1.5, 1.5, 500)
M_vz = np.linspace(-1.5, 1.5, 500)
img = [[Pi_0(0, vy, vz) for vy in M_vy] for vz in M_vz]
fig = plt.figure()
plt.imshow(img, extent=[min(M_vy), max(M_vy), min(M_vz), max(M_vz)], aspect='equal', origin='lower')
plt.xlabel("$v_y$ (m/s)")
plt.ylabel("$v_z$ (m/s)")    
plt.colorbar()
#plt.savefig("Pi_0.pdf")
plt.show()


"""Calculation of the quantum amplitudes cn"""
def c0(qz, n):
    h2 = h + 2*1j*qz*zeta**2/hbar
    return (8*np.pi*zeta**2)**(1/4)/(np.sqrt(lg)*airy(-Lambda[n])[1])*airy(h2/lg-Lambda[n]+(zeta/lg)**4)[0]*np.exp((zeta/lg)**2*(h2/lg-Lambda[n]+2/3*(zeta/lg)**4)-(qz*zeta/hbar)**2)
M_qz = np.linspace(-qe, qe, 10000)
start_time = time.time()
c = np.array([[c0(qz, n) for qz in M_qz] for n in range(GQS)])
print("Time execution cn coefficients: %s seconds ---" % round(time.time() - start_time))   
#np.save("c.npy", c)

"""Calculation of the number of atoms detected"""
def c_n_horizontal_polarization(theta, GQS):
    qz = qe*np.cos(theta)
    A = sum([abs(c0(qz, n))**2 for n in range(GQS)])
    return np.pi*3/(4*np.pi)*np.sin(theta)**3*A
N_atoms = 1000*integrate.quad(c_n_horizontal_polarization, 0, np.pi, args=(GQS))[0]
N_atoms = int(N_atoms)
print("N_atoms =", N_atoms)


"""Computation of the Fourier transform of \chi_n"""
freq = np.linspace(-25*pg, 25*pg, 60001)
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
A = np.array([Chi_FT(n) for n in range(1000)])
print("Time execution Fourier transform: %s seconds ---" % round(time.time() - start_time))    
#np.save('Chi-FT_25pg_60000_n=1000.npy', A)

Chi_FT = np.load('Chi-FT_25pg_60000_n=1000.npy')/np.sqrt(2*np.pi*hbar)
M_pz = np.linspace(-25*pg, 25*pg, len(Chi_FT[0]))


"""For the discretization of the integrals in the function Pi_d, done with Riemann sum"""
theta_discretization = 50
theta_e = np.linspace(0, np.pi, theta_discretization)
dtheta = theta_e[1] - theta_e[0]
phi_e = np.linspace(0, 2*np.pi, theta_discretization)
dphi = phi_e[1] - phi_e[0]
polar = np.array([0, np.sin(theta_n), np.cos(theta_n)])

"""Parallelized calculation of the velocity distribution at the end of the mirror"""
@nb.njit(parallel=True)
def Pi_d(px, py, pz, d):
    if px**2+py**2==0:
        return 0
    t = m*d/np.sqrt(px**2+py**2) # time spent above the mirror
    f = np.zeros((theta_discretization-1, theta_discretization-1))
    for i in nb.prange(theta_discretization-1):
        for j in range(theta_discretization-1):
            theta = theta_e[j]
            phi = phi_e[i]
            qe_unit = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])    
            qx = -qe*qe_unit[0]
            qy = -qe*qe_unit[1]
            qz = -qe*qe_unit[2]              
            index_qz = np.argmin(np.abs(M_qz-qz))          
            index_pz = np.argmin(np.abs(M_pz-pz))
            A = 0
            for n in range(GQS):
                #value = (Chi_FT[n][index_pz+1]-Chi_FT[n][index_pz-1])/(M_pz[index_pz+1]-M_pz[index_pz-1])*(pz-M_pz[index_pz])+Chi_FT[n][index_pz]
                value = Chi_FT[n][index_pz]
                A += c[n][index_qz]*value*np.exp(-1j*Lambda[n]*t/tg)
            f[i][j] = np.sin(theta)*3/(4*np.pi)*(np.vdot(qe_unit, polar))**2*1/(2*np.pi*Delta_p**2)*np.exp(-((px-qx)**2+(py-qy)**2)/(2*Delta_p**2))*np.abs(A)**2
    return dtheta*dphi*np.sum(f)

"""Plot of the velocity distribution at the end of the mirror"""
start_time = time.time()
M_vz = np.linspace(-5e-2, 5e-2, 1000)
f = [Pi_d(0, qe, m*vz, d) for vz in M_vz]
plt.figure()
plt.plot(M_vz*100, f) 
plt.xlabel("$v_z$ (cm/s)")  
plt.title('$\Pi_t~(kg^{-3}m^{-3}s^3)$')
#plt.savefig("Pi_t-1D.pdf")
plt.show()
print("Time execution Plot Pi_d-1D: %s seconds ---" % round(time.time() - start_time)) 

