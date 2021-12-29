#Calculates the radio photons from accreting PBHs. If you use this code please consider citing arXiv:2110.11975
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
ρdm=3*Ho**2/(8*np.pi*6.67e-11*(3.0857e19)**2)*(Ω_m-Ω_B) #Dark matter density in SI units.

def E2nu(E):
    return E*1e-6*1.6e-19*1e-9/(6.634e-34)

def nu2E(nu):
    return nu*6.634e-34/(1e-6*1.6e-19*1e-9)


def H(Z):       #Hubble function in units of 1 km/s/Mpc. 'Z' stands for (1+z)
        return Ho*(Ω_r*Z**4+Ω_m*Z**3+Ω_Λ)**0.5


A=0.019*1e-2*1e-4   #This is the product of fduty, (fx. λ)^0.86 & fpbh. This value of 'A' generates just the right amount of ERB
#to explain the EDGES signal, which is ~5% of the ARCADE2/LWA1 limit. See arXiv:2107.02190.

n=-2.6              #Spectral index of the ERB

def ERB(E):
        return Tγo+0.0541*24.1*(E/1.282)**-2.6

def ERB1(E):
        return Tγo+24.1*(E/1.282)**-2.6

#Uncomment the following 4 lines to calculate the specific intensity and equivalent brightness temperature at the wavelength of 21 cm. 
#J_21cm=A*ρdm*4.12e46*scint.quad(lambda Z:Z**(n+1)/H(Z),1,1000)[0]
#print('J(E=5.9 μeV)=',J_21cm,'s^-1.m^-2.sr^-1')
#print('T =',9.5e-61*J_21cm/(5.9e-6*1.6e-19)**2,' K')

E=np.logspace(-1,2)
J=(E/5.9)**(n+2)*A*ρdm*4.12e46*scint.quad(lambda Z:Z**(n+1)/H(Z),1,np.inf)[0]
T=9.5e-61*J/(E*1e-6*1.6e-19)**2 #This excess radio temperature explains the 5% limit of ARCADE2/LWA1, as explained in line 26 & 27.

fig,ax=plt.subplots()

ax.loglog(E,ERB1(E),color='b',lw=1.5,label='ARCADE2/LWA1',ls=':')
ax.loglog(E,Tγo+19.956*T,color='b',lw=1.5,label=r'$(\lambda, f_{\mathrm{duty}})=(0.5,0.05)$')
ax.loglog(E,ERB(E),color='r',lw=1.5,label='Mittal et al. (2021)',ls=':')
ax.loglog(E,Tγo+T,color='r',lw=1.5,label=r'$(\lambda, f_{\mathrm{duty}})=(0.1,0.01)$')

ax.axhline(y=Tγo,color='k',ls='--',lw=1.5, label='CMB')
ax.axvspan(0.1, 43.4,color='lightgrey',alpha=0.5)
ax.legend(fontsize=16,loc=1,frameon=False)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('$E\,(\mu$eV)',fontsize=22)
ax.set_ylabel('$T_{\mathrm{r}}$ (K)',fontsize=22)

secax = ax.secondary_xaxis('top', functions=(E2nu,nu2E))
secax.set_xlabel(r'$\nu$ (GHz)',fontsize=22, labelpad=10)
secax.tick_params(which='major', labelsize=18)

ax.tick_params(axis='both', which='major', labelsize=16)
ax.minorticks_on()
#ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.yaxis.set_ticks_position('both')

plt.xlim([0.1,1e2])
plt.ylim([1,3e4])
plt.show()
