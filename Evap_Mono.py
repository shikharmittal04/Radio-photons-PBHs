#This is for masses for which I have flux very low energies as well.
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
from scipy import interpolate
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.lss import peaks
my_cosmo = {'flat': True, 'H0': 67.4, 'Om0': 0.315, 'Ob0': 0.049, 'sigma8': 0.811, 'ns': 0.965,'relspecies': False,'Tcmb0': 2.725}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)

Ho=67.4
Ω_m=0.315
Ω_B=0.049
Ω_Λ=0.685
Tγo=2.725
Yp=0.245
Ω_r=7.53e-3*Tγo**4/Ho**2
ρdm=3*Ho**2/(8*np.pi*6.67e-11*(3.0857e19)**2)*(Ω_m-Ω_B)

def H(Z):       
        return Ho*(Ω_r*Z**4+Ω_m*Z**3+Ω_Λ)**0.5


E=np.load('E-e28.npy')
#photon=np.load('Flux-e20-27.npy')[:,11]
photon=np.load('Flux-e28.npy')

tck = interpolate.splrep(E,photon)

def ene_flx(E,Z):
    d2NdEdt_rs=interpolate.splev(E*Z,tck)
    return E*d2NdEdt_rs

Z=np.linspace(1,1e3,10000)
n=920
J=np.zeros(n)
M=1e25#mass of black hole in kg
for i in range(n):
        J[i]=6.28*(Ω_m-Ω_B)*scint.trapz(ene_flx(E[i],Z)/H(Z),Z)*1/M

tck1 = interpolate.splrep(1e15*E[0:n],J)
J0=interpolate.splev(5.9,tck1)

print('J(E=5.9 μeV)=',J0)
print('T =',9.5e-61*J0/(5.9e-6*1.6e-19)**2,' K')

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rc('text.latex', preamble=r'\usepackage{siunitx}')

fig,ax=plt.subplots()

ax.loglog(1e15*E[0:n],J,'b')
ax.minorticks_on()
ax.axvline(x=5.9,color='r',ls='--')
#ax.text(8e-6,1e-38,r'$T_{\mathrm{r}}=1.75\times10^{-48}$ K',fontsize=20)
#ax.set_xlabel('$E\,(\si{\micro\electronvolt}$)',fontsize=22)
ax.set_ylabel('$J$ (s$^{-1}$ m$^{-2}$ sr$^{-1}$)',fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=18)
#ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#ax.set_aspect('equal', adjustable='box')
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')

#plt.xlim([0.1,1e2])
#plt.ylim([1e-23,1e-20])
#plt.xlim([1,1e13])
#plt.ylim([1e-23,1e-20])
plt.show()
