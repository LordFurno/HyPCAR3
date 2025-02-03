import itertools
from pyDOE import lhs
import numpy as np
import random
from astropy import constants as const
import scipy.stats as ss
import scipy.constants as sc
import scipy.special   as sp
import time

#xi and PT_line from BART
def xi(gamma, tau):
    """
    Calculate Equation (14) of Line et al. (2013) Apj 775, 137

    Parameters:
    -----------
    gamma: Float
        Visible-to-thermal stream Planck mean opacity ratio.
    tau: 1D float ndarray
        Gray IR optical depth.
    """
    return (2.0/3) * \
            (1 + (1./gamma) * (1 + (0.5*gamma*tau-1)*np.exp(-gamma*tau)) +
            gamma*(1 - 0.5*tau**2) * sp.expn(2, gamma*tau)              )
def PT_line(pressure, kappa,  gamma1, gamma2, alpha, beta, R_star,   T_star, T_int,  sma,    grav,  T_int_type):
    '''
    Generates a PT profile based on input free parameters and pressure array.
    If no inputs are provided, it will run in demo mode, using free
    parameters given by the Line 2013 paper and some dummy pressure
    parameters.

    Inputs
    ------
    pressure: 1D float ndarray
        Array of pressure values in bars.
    kappa : float, in log10. Planck thermal IR opacity in units cm^2/gr
    gamma1: float, in log10. Visible-to-thermal stream Planck mean opacity ratio.
    gamma2: float, in log10. Visible-to-thermal stream Planck mean opacity ratio.
    alpha : float.           Visible-stream partition (0.0--1.0).
    beta  : float.           A 'catch-all' for albedo, emissivity, and day-night
                            redistribution (on the order of unity)
    R_star: Float
        Stellar radius (in meters).
    T_star: Float
        Stellar effective temperature (in Kelvin degrees).
    T_int:  Float
        Planetary internal heat flux (in Kelvin degrees).
    sma:    Float
        Semi-major axis (in meters).
    grav:   Float
        Planetary surface gravity (at 1 bar) in cm/second^2.
    T_int_type: string.
        Method for determining `T_int`: 'const' (for a supplied constant value)
                                        'thorngren' (to use Thorngren et al. 2019)

    Returns
    -------
    T: temperature array
    Developers:
    -----------
    Madison Stemm      astromaddie@gmail.com
    Patricio Cubillos  pcubillos@fulbrightmail.org
    '''
    # Convert kappa, gamma1, gamma2 from log10
    kappa  = 10**(kappa )
    gamma1 = 10**(gamma1)
    gamma2 = 10**(gamma2)

    if T_int_type == 'thorngren':
        # Planetary internal temperature (Thorngren et al. 2019)
        # Hard-coded values are fitted parameters!
        T_eq  = (R_star/(2.0*sma))**0.5 * T_star
        F     = 4.0 * sc.Stefan_Boltzmann * T_eq**4
        T_int = 1.24 * T_eq * np.exp(-(np.log(F) - 0.14)**2 / 2.96)

    # Stellar input temperature (at top of atmosphere):
    T_irr = beta * (R_star / (2.0*sma))**0.5 * T_star

    # Gray IR optical depth:
    tau = kappa * (pressure*1e6) / grav # Convert bars to barye (CGS)

    xi1 = xi(gamma1, tau)
    xi2 = xi(gamma2, tau)

    # Temperature profile (Eq. 13 of Line et al. 2013):
    temperature = (0.75 * (T_int**4 * (2.0/3.0 + tau) +
                            T_irr**4 * (1-alpha) * xi1 +
                            T_irr**4 * alpha     * xi2 ) )**0.25
    return temperature



def generatePlanet(luminosity,sma):
    '''
    This function will generate planet parameters

    Inputs
    ------
    luminosity: Star luminosity
    sma: Semi Major Axis

    Returns
    -------
    planetRad: Planet radius
    planetMass: Planet mass
    density: Planet density
    grav: Planet surface gravity
    '''
    canHoldAtmosphere=False
    while not canHoldAtmosphere:
        planetRad=np.random.uniform(0.5,1.6)#In Earth radii
        # Calcuate mass from radius using relation from
        #Sotin et al 2007, Icarus, 
        #"Mass-radius curve for extrasolar Earth-like planets and ocean planets"
        #Including a factor of +- 2% to account for the over/under shoot in paper
        if planetRad>1:
            planetMass=planetRad**(1./0.274) * np.random.uniform(0.98, 1.02)
        else:
            planetMass=planetRad**(1./0.306) * np.random.uniform(0.98, 1.02)
        #Calculates density, mass/volume
        #In g/cm^3
        density=planetMass*const.M_earth.cgs.value / (4./3.*np.pi*(planetRad * const.R_earth.cgs.value)**3)

        #Calculates surface gravity
        #In cm/s^2
        grav=const.G.cgs.value*planetMass*const.M_earth.cgs.value / (planetRad*const.R_earth.cgs.value)**2

        #Calculates surface escape gravity
        #In km/s^2
        vesc = (2.*const.G.value*planetMass*const.M_earth.value / (planetRad*const.R_earth.value))**0.5

        #Calculate planet insolation, relative to Earth
        insol=luminosity/const.L_sun.value/sma**2

        #Rough approx of 'cosmic shoreline' eqn in Zahnle & Catling 2016
        #1e-6 Insol : 0.2 vesc, 1e4 Insol : 70 vesc -- straight line on loglog
        #Slope of cosmic shoreline
        shoreline = np.log10(1e4 / 1e-6)/np.log10(70. / 0.2)

        #Calculate shoreline insolation, pinsol, value for the planet's vesc 
        #to see if it can hold an atmosphere
        #Requires the actual insolation, Insol, to be less than pinsol
        pinsol=1e4 * (vesc / 70.)**shoreline
        if insol<pinsol:
            canHoldAtmosphere=True
    return planetRad,planetMass,density,grav

def generatePT(kappa,gamma1,gamma2,alpha,beta,pmin,pmax,starRad,starTemp,sma,grav):
    '''
    This function generatures a pressure-temperature (PT) profile for given
    input parameters using the model of Line et al. (2013).

    Note that this uses PT.py from BART, the Bayesian Atmospheric
    Radiative Transfer code, which has an open-source, reproducible-research license.

    Inputs
    ------
    kappa:   Planck thermal IR opacity (in units cm^2/gr)
    gamma1:  Visible-to-thermal stream Planck mean opacity ratio.
    gamma2:  Visible-to-thermal stream Planck mean opacity ratio.
    alpha:         Visible-stream partition (0.0--1.0).
    beta:          A 'catch-all' for albedo, emissivity, and day-night
                      redistribution (on the order of unity).
    pmin:          Minimum pressure at the top of the atmosphere (in bars).
    pmax:          Pressure at the surface of the planet (in bars).
    starRad:         Radius of the host star (in solar radii).
    starTemp:         Temperature of the host star (in Kelvin).
    sma:           Semimajor axis of planet (in AU).
    grav:      Planetary gravity at 1 bar of pressure (in cm/s^2).

    Returns
    -------
    ptProfile: 2D array containing the pressure (equally spaced in
                logspace) and temperature at each pressure.
                PTprof[i] gives the i-th layer's [pressure, temperature]
                PTprof[:,0] gives the pressure at all layers.
                PTprof[:,1] gives the temperature at all layers.
    '''

    pressure=np.logspace(np.log10(pmax),np.log10(pmin),50)#50 layers
    #PT_line(pressure, kappa,  gamma1, gamma2, alpha, beta, R_star,   T_star, T_int,  sma,    grav,  T_int_type):
    temperatures=PT_line(pressure, kappa,gamma1,gamma2,alpha,beta, starRad * const.R_sun.value, starTemp , 0., sma*const.au.value, grav,'const')

    ptProfile=np.zeros((50, 2), dtype=float)
    ptProfile[:,0]=pressure
    ptProfile[:,1]=temperatures

    return ptProfile


'''
https://arxiv.org/pdf/2010.12241
Under ideal gas conditions and mixing, partial pressure equates to molecular abundances. 
This assumption is used here.
Type A atmopsheres:
    Hydrogen-rich
    Contains H2O, CH4, NH3, H2/N2
    Lacks CO2, O2
    A1 atmospheres:
        Mainly contains: H2O, CH4, NH3, and H2
        Lacks: CO2, O2
        H > 2O + 4C
        3N < H - 2O -4C

        D = H - N - 2C
        H2O = 2O / D
        NH3 = 2N / D
        CH4 = 2C / D
        H2 = (H - 2O - 4C - 3N) / D
    A2 atmospheres:
        Mainly contains: H2O, CH4, NH3, N2
        Lacks: CO2, O2
        H > 2O + 4c
        3N > H - 2O - 4C
         
        D = H + 2C + 3N + 4O
        H2O = 6O / D
        NH3 = (2H - 8C - 4O) / D
        CH4 = 6C / D
        N2 = (3N + 4C + 2O - H) / D
Type B atmospheres:
    Oxygen rich
    Mainly contain O2, N2, CO2, H2O
    Lacks NH3, H2
    2O > H + 4C
    D = H + 2O + 2N
    H2O = 2H / D
    N2 = 2N / D
    CO2 = 4C / D
    O2 = (2O - H - 4C) /D
Type C atmospheres:
    Contains H2O, CO2, CH4, N2
    Lacks NH3, H2, O2 

    H + C + O + N = 1
    This is unlike the others, which their abundances defined in repsect to hydrogen
    This is a hydrogen-poor atmosphere

    Side conditions, so no negative results
    O > 0.5H + 2C -> O2-rich with no CH4
    H > 2O + 4C -> H2 ->H2-rich with no CO2
    C > 0.25H + 0.5O -> graphite condensation with no H2O

    H2O = (H + 2O - 4C) / (H + 2O + 2N)
    CH4 = (H - 2O + 4C) / (2H + 4O + 4N)
    CO2 = (2O + 4C - H) / (2H + 4O + 4N)
    N2 = 2N / (H + 2O + 2N)

Based on solar elemental abundances:
https://arxiv.org/pdf/0909.0948

The ranges for the elemental molecules will be:

Hydrogen will be 1.0
Everything is based off of that
O_min, O_max = 1e-3, 1e-1
C_min, C_max = 1e-4, 1e-2
N_min, N_max = 1e-5, 1e-3

For C-type atmospheres: 
Just randomly sample O,C,N,H and normalize
Check if it fulfills the conditions and then calculate abundances
'''
#Molecule order will be:
#O2, N2, H2, CO2, H2O, CH4, NH3
def calculateMoleculeAbundances(atmosphereType):
    #Molecule order will be:
    #O2, N2, H2, CO2, H2O, CH4, NH3
    abundances=[0.0]*7

    #Sampling ranges for elemental abundances. Based on solar abundances with some variability.
    O_min, O_max = 1e-3, 1e-1
    C_min, C_max = 1e-4, 1e-2
    N_min, N_max = 1e-5, 1e-3

    O_mean=(np.log(O_min)+np.log(O_max))/2
    O_std=(np.log(O_max)-np.log(O_min))/4

    C_mean=(np.log(C_min)+np.log(C_max))/2
    C_std=(np.log(C_max)-np.log(C_min))/4

    N_mean=(np.log(C_min)+np.log(C_max))/2
    N_std=(np.log(N_max)-np.log(N_min))/4


    if atmosphereType=="A1":
        '''
        Mainly contains: H2O, CH4, NH3, and H2
        Lacks: CO2, O2
        H > 2O + 4C
        3N < H - 2O -4C

        D = H - N - 2C
        H2O = 2O / D
        NH3 = 2N / D
        CH4 = 2C / D
        H2 = (H - 2O - 4C - 3N) / D
        '''
        #Draw from log-normal distrbution. This makes sense, as abundances are on logarithmic scale
        #Hydrogen is assumed to be 1.0, as that is baseline
        
        a,b,c,d=np.random.exponential(scale=1.0,size=4)
        total=a+b+c+d
        H, O, C, N = a/total, b/total, c/total, d/total

        D = H - N - 2*C

        #Check A1 atmosphere conditions

        while not ((H > 2*O + 4*C) and (3*N < H - 2*O - 4*C) and (D > 0)):
            a,b,c,d=np.random.exponential(scale=1.0,size=4)
            total=a+b+c+d
            H, O, C, N = a/total, b/total, c/total, d/total
            D = H - N - 2*C

       
        H2O = 2*O / D
        NH3 = 2*N / D
        CH4 = 2*C / D
        H2 = (H - 2*O - 4*C - 3*N) / D

        #Percentages
        abundances[2]=H2
        abundances[4]=H2O
        abundances[5]=CH4
        abundances[6]=NH3
        return abundances
    elif atmosphereType=="A2":
        '''
        Mainly contains: H2O, CH4, NH3, N2
        Lacks: CO2, O2
        H > 2O + 4c
        3N > H - 2O - 4C
         
        D = H + 2C + 3N + 4O
        H2O = 6O / D
        NH3 = (2H - 8C - 4O) / D
        CH4 = 6C / D
        N2 = (3N + 4C + 2O - H) / D
        '''
        a,b,c,d=np.random.exponential(scale=1.0,size=4)
        total=a+b+c+d
        H, O, C, N = a/total, b/total, c/total, d/total
        D = H + 2*C + 3*N + 4*O

        #Check A2 atmosphere conditions
        while not ((H > 2*O + 4*C) and (3*N > H - 2*O - 4*C) and (D > 0)):
            a,b,c,d=np.random.exponential(scale=1.0,size=4)
            total=a+b+c+d
            H, O, C, N = a/total, b/total, c/total, d/total
            D = H + 2*C + 3*N + 4*O

        H2O = 6*O / D
        NH3 = (2*H - 8*C - 4*O) / D
        CH4 = 6*C / D
        N2 = (3*N + 4*C + 2*O - H) / D
        
        #Percentages
        abundances[1]=N2
        abundances[4]=H2O
        abundances[5]=CH4
        abundances[6]=NH3
        return abundances
    
    elif atmosphereType=="B":
        '''
        Oxygen rich
        Mainly contain O2, N2, CO2, H2O
        Lacks NH3, H2
        2O > H + 4C
        D = H + 2O + 2N
        H2O = 2H / D
        N2 = 2N / D
        CO2 = 4C / D
        O2 = (2O - H - 4C) /D
        '''
        a,b,c,d=np.random.exponential(scale=1.0,size=4)
        total=a+b+c+d
        H, O, C, N = a/total, b/total, c/total, d/total
        D = H + 2*O + 2*N
        while not ((2*O > H + 4*C) and (D>0)):
            a,b,c,d=np.random.exponential(scale=1.0,size=4)
            total=a+b+c+d
            H, O, C, N = a/total, b/total, c/total, d/total
            D = H + 2*O + 2*N
        
        H2O = 2*H / D
        N2 = 2*N / D
        CO2 = 4*C / D
        O2 = (2*O - H - 4*C) / D

        #O2, N2, H2, CO2, H2O, CH4, NH3
        abundances[0]=O2
        abundances[1]=N2
        abundances[3]=CO2
        abundances[4]=H2O
        return abundances

    elif atmosphereType=="C":
        '''
        Contains H2O, CO2, CH4, N2
        Lacks NH3, H2, O2 

        H + C + O + N = 1
        This is unlike the others, which their abundances defined in repsect to hydrogen
        This is a hydrogen-poor atmosphere

        Side conditions, so no negative results
        O > 0.5H + 2C -> O2-rich with no CH4
        H > 2O + 4C -> H2 ->H2-rich with no CO2
        C > 0.25H + 0.5O -> graphite condensation with no H2O

        H2O = (H + 2O - 4C) / (H + 2O + 2N)
        CH4 = (H - 2O + 4C) / (2H + 4O + 4N)
        CO2 = (2O + 4C - H) / (2H + 4O + 4N)
        N2 = 2N / (H + 2O + 2N)
        '''
        while True:
            #Sample 4 random numbers and normalize
            a,b,c,d=np.random.exponential(scale=1.0,size=4)
            total=a+b+c+d
            H, O, C, N = a/total, b/total, c/total, d/total

            D1 = H + 2*O + 2*N
            D2 = 2*H + 4*O + 4*N
            if D1<=0 or D2<=0:
                continue
                

            H2O = (H + 2*O - 4*C) / D1
            CH4 = (H - 2*O + 4*C) / D2
            CO2 = (2*O + 4*C - H) / D2
            N2 = 2*N / D1

            #Check for side conditions
            if H2O < 0 or CH4 < 0 or CO2 < 0 or N2 < 0:
                continue 
            
            #Valid sample found
            #calculate abundances


          

            abundances[1]=N2
            abundances[3]=CO2
            abundances[4]=H2O
            abundances[5]=CH4

            return abundances
start=time.time()
a1=calculateMoleculeAbundances("A1")
a2=calculateMoleculeAbundances("A2")
b=calculateMoleculeAbundances("B")
c=calculateMoleculeAbundances("C")
#O2, N2, H2, CO2, H2O, CH4, NH3
# test=set([])
# counter=0
# for i in range(1000):
#     b=calculateMoleculeAbundances("B")
#     if b.index(max(b))==1:
#         print(max(b))
#         counter+=1
#     test.add(b.index(max(b)))
# print(test)
# print(counter)

print(f"Abundances for A1-Type: {a1}")

print(f"Abundances for A2-Type: {a2}")

print(f"Abundances for B-Type: {b}")

print(f"Abundances for C-Type: {c}")


print(time.time()-start)