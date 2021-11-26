#Calculates the radio background from low mass evaporating PBHs. If you use this code please consider citing arXiv:2110.11975
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
from scipy import interpolate

Ho=67.4
Ω_m=0.315
Ω_B=0.049
Ω_Λ=0.685
Tγo=2.725
Yp=0.245
Ω_r=7.53e-3*Tγo**4/Ho**2
ρdm=3*Ho**2/(8*np.pi*6.67e-11*(3.0857e19)**2)*(Ω_m-Ω_B)

#The following is the Hubble function in units of 1 km s^-1 Mpc^-1. Everywhere in these codes 'Z' stands for (1+z).
def H(Z):       
        return Ho*(Ω_r*Z**4+Ω_m*Z**3+Ω_Λ)**0.5

#Important: the following 2 inputs are obtained from the C code BlackHawk. 'E.npy' contains the energies in GeV
#and 'flux.npy' contains the corresponding flux values in units of GeV^-1 s^-1. An example is provided; for mass 10^25 kg. Each array has 1000 elements.
E=np.load('E.npy')
flux=np.load('flux.npy')

tck = interpolate.splrep(E,flux)

#The following function returns E.F[E.(1+z)] in units of s^-1 - see notation in equation (6) from the paper.
def ene_flx(E,Z):       
    d2NdEdt_rs=interpolate.splev(E*Z,tck)
    return E*d2NdEdt_rs

Z=np.linspace(1,1e3,10000)
n=920           #Choose any value <=1000 for this. 
J=np.zeros(n)
M=1e25          # Enter the mass of black hole in kg

#The following is final quantity of our interest (equation 15 from the paper). It gives the specific intensity in units of (s^-1.m^-2.sr^-1).
for i in range(n):
        J[i]=6.28*(Ω_m-Ω_B)*scint.trapz(ene_flx(E[i],Z)/H(Z),Z)*1/M


# Uncomment the following 4 lines to calculate the specific intensity and equivalent brightness temperature at the wavelength of 21 cm.  
#tck1 = interpolate.splrep(1e15*E[0:n],J)
#J_21cm=interpolate.splev(5.9,tck1)
#print('J(E=5.9 μeV)=',J_21cm)
#print('T =',9.5e-61*J_21cm/(5.9e-6*1.6e-19)**2,' K')


fig,ax=plt.subplots()

ax.loglog(1e15*E[0:n],J,'b')
ax.minorticks_on()
ax.axvline(x=5.9,color='r',ls='--')
ax.set_xlabel('$E\,\mu$eV)',fontsize=22)
ax.set_ylabel('$J$ (s$^{-1}$ m$^{-2}$ sr$^{-1}$)',fontsize=22)
plt.show()
