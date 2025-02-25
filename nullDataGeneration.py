
from pyDOE import lhs
import numpy as np
import random
from astropy import constants as const
import scipy.stats as ss
import scipy.constants as sc
import scipy.special   as sp
import time
import os
from processing import callPSG
#Set the target molecules "O2","N2","H2","CO2","H2O","CH4","NH3", to be very low
#This will still be the one-hot vector. The other molecules will be in the background so-to-speak 
#This means that it will still affect the transmission spectra, just won't show up in the one-hot vector
#This means I have to be careful when getting the label for the detection model
#Lets have 3 background molecules:
#CO
#SO2
#H2S

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

def lhsSampling(params,nSamples):
    '''
    This function will perform the Latin Hypercube Sampling and scale the data.

    Inputs
    ------
    params: The list of parameters that will be sampled
    nSamples: Number of samples to be generated

    Returns
    -------
    scaledSamples: A matrix containing the samples
    '''
    lhsSamples=lhs(len(params),nSamples)

    scaledSamples=np.zeros_like(lhsSamples)

    for i, (key,(minVal,maxVal)) in enumerate(params.items()):
        scaledSamples[:,i]=lhsSamples[:,i]*(maxVal-minVal)+minVal
    return scaledSamples

def createConfigFile(independant,dependant,moleculeAbundances,starType,filePath):
    '''
    This function will create a config file for the example exoplanet generated. 
    This is the file that will be passed to PSG when actually generating the transmittance data.

    Inputs
    ------
    independant: The independant parameters that were generated.
                 This includes starRad, starTemp, kappa, gamma1, gamm2, alpha, albedo,distance
    dependant: The independant parameters that were generated.
                This includes semiMajorAxis, planetRad, planetMass, density, grav, surfTemp, surfPres, ptProfile
    moleculeAbundances: The abundances of eaech molecule: O2, N2, H2, CO2, H2O, CH4, NH3
    starType: The host star's type (g,m,k)
    filePath: The filepath of this config file
    
    
    Returns
    -------
    None
    '''
    
    #Independant order:
    #StarRad,starTemp,Kappa,Gamma1,Gamma2,alpha,Albedo,Distance, molecule1,molecule2....

    starRad,starTemp,kappa,gamma1,gamma2,alpha,albedo,dist=independant
    
    semiMajorAxis,planetRad,planetMass,planetDensity,planetGrav,surfTemp,surfPres,PTprofile=dependant
    
    #g/mol
    #O2, N2, H2, CO2, H2O, CH4, NH3

    abundanceDictionary={}
    molecules=["O2","N2","H2","CO2","H2O","CH4","NH3","CO","SO2","H2S"]
    for i in range(len(moleculeAbundances)):
        abundanceDictionary[molecules[i]]=moleculeAbundances[i]

    moleculeWeights={"O2":31.999, "N2":28.02, "H2":2.016,"CO2":44.01, "H2O":18.01528,"CH4":16.04,"NH3":17.03052,"CO":28.01,"SO2":64.066,"H2S":32.082}#g/mol
    averageWeight=0
    for molecule in abundanceDictionary:
        averageWeight+=moleculeWeights[molecule]*abundanceDictionary[molecule]
    
    HITRANValues={"O2":"HIT[7]","N2":"HIT[22]","H2":"HIT[45]","CO2":"HIT[2]","H2O":"HIT[1]","CH4":"HIT[6]","NH3":"HIT[11]","CO":"HIT[5]","SO2":"HIT[9]","H2S":"HIT[31]"}
    
    #Copys the config file
    lines=[]
    # configTemplatePath=r"C:\Users\Tristan\Downloads\HyPCAR3\configTemplate.txt"
    configTemplatePath="/home/tristanb/projects/def-pjmann/tristanb/configTemplate.txt"
    with open(configTemplatePath) as template:
        for line in template:
            lines.append(line)

    #System information
    lines[3]="<OBJECT-DIAMETER>"+str(2*planetRad*const.R_earth.value/1000)+"\n" #Planet diameter [km]
    lines[4]="<OBJECT-GRAVITY>"+str(planetDensity)+"\n" #Planet density [g/cm3], the gravity part is just straight up wrong
    lines[7]="<OBJECT-STAR-DISTANCE>"+str(semiMajorAxis)+"\n" #Semimajor axis [AU]
    lines[11]="<OBJECT-STAR-TYPE>"+str(starType)+"\n" #Stellar class
    lines[12]="<OBJECT-STAR-TEMPERATURE>"+str(starTemp)+"\n" #Star temperature [K]
    lines[13]="<OBJECT-STAR-RADIUS>"+str(starRad)+"\n" #Stellar radius [Rsun]
    lines[23]="<GEOMETRY-OBS-ALTITUDE>"+str(dist)+"\n" #Distance to system

    #Atmosphere information
    lines[42]="<ATMOSPHERE-NGAS>"+str(len(moleculeAbundances))+"\n" #Number of gases are in the atmosphere
    lines[43]="<ATMOSPHERE-GAS>"+",".join(molecules)+"\n" #What gases are in the atmosphere
    lines[44]="<ATMOSPHERE-TYPE>"+",".join(HITRANValues[mol] for mol in molecules)+"\n" #HITRAN values for each gas
    lines[45]="<ATMOSPHERE-ABUN>"+"1,"*(len(moleculeAbundances)-1)+"1"+"\n" #Molecule abunadnces. They're all 1, because abundances are defined in vertical profile
    lines[46]="<ATMOSPHERE-UNIT>"+"scl,"*(len(moleculeAbundances)-1)+"scl"+"\n" #Abundance unit
    lines[49]="<ATMOSPHERE-WEIGHT>"+str(averageWeight)+"\n" #Molecule weight of atmosphere g/mol
    lines[50]="<ATMOSPHERE-PRESSURE>"+str(surfPres)+"\n" #Planetary surface pressure bars
    lines[52]="<ATMOSPHERE-LAYERS-MOLECULES>"+",".join(molecules)+"\n" #Molecule in vertical profile

    #Atmosphere layers
    #Starts at line 54

    for i in range(50):
        atmosphereInfo=",".join(map(str,PTprofile[i]))+","+",".join(map(str,list(abundanceDictionary.values())))
        lines[54+i]="<ATMOSPHERE-LAYER-"+str(i+1)+">"+atmosphereInfo+"\n"

    #Surface information
    lines[112]="<SURFACE-TEMPERATURE>"+str(surfTemp)+"\n" 
    lines[113]="<SURFACE-ALBEDO>"+str(albedo)+"\n"
    lines[114]="<SURFACE-EMISSIVITY>"+str(1.-albedo)+"\n"

    #Write to new config file
    with open(filePath,"w") as f:
        f.writelines(lines)

def calculateMoleculeAbundances():
    knownValues=np.random.uniform(0,0.001,size=7)#Known moleecule abundancese
    totalKnown=np.sum(knownValues)
    remainingAbundance=1-totalKnown

    backgroundAbundances=np.random.dirichlet(np.ones(3))*remainingAbundance

    return np.array(list(knownValues)+list(backgroundAbundances))




if __name__=="__main__":
    start=time.time()
    starTypes=["G","M","K"]
    molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]
    atmosphereTypes=["None"]
    seed=42
    random.seed(seed)
    np.random.seed(seed)

    for aType in ["None"]:
        folderPath="/home/tristanb/scratch/data"+f"/{aType}"

        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

    #Creates folder for the config files
    # configFolder=r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles"
    configFolder="/home/tristanb/scratch/configFiles"
    if not os.path.exists(configFolder):
        os.makedirs(configFolder)

    run=True
    for atmosphereType in atmosphereTypes:
        gStarParamRanges = {
        'starRad': (0.8, 1.3),
        'starTemp': (5000, 6000),
        'Kappa': (-3.5, -2.0),
        'Gamma1': (-1.5,  1.1),
        'Gamma2': (-1.5,  0.),
        'alpha': ( 0.,   1.),
        'Albedo':(0.1, 0.8),
        'Distance': (1.3,15.)}
        mStarParamRanges = {
            'starRad': (0.14, 0.55),
            'starTemp': (3000, 3800),
            'Kappa': (-3.5, -2.0),
            'Gamma1': (-1.5,  1.1),
            'Gamma2': (-1.5,  0.),
            'alpha': ( 0.,   1.),
            'Albedo':(0.1, 0.8),
            'Distance': (5.,25.)}
        kStarParamRanges = {
            'starRad': (0.6, 0.95),
            'starTemp': (3800, 5300),
            'Kappa': (-3.5, -2.0),
            'Gamma1': (-1.5,  1.1),
            'Gamma2': (-1.5,  0.),
            'alpha': ( 0.,   1.),
            'Albedo':(0.1, 0.8),
            'Distance': (1.3,15.)}

        nSamples=20000

        gStarSamples=lhsSampling(gStarParamRanges,nSamples)
        mStarSamples=lhsSampling(mStarParamRanges,nSamples)
        kStarSamples=lhsSampling(kStarParamRanges,nSamples)

        #Each sample is in form of [StarRad,starTemp,Kappa,Gamma1,Gamma2,alpha,Albedo,Distance,molecule1,molecule2...]
        
        #Adjusting star radius based on their temperature:
        for index,starSample in enumerate([gStarSamples,mStarSamples,kStarSamples]):
            for i,sample in enumerate(starSample):
                if index==0:#G type star
                    if sample[1]>5500:
                        sample[0]=np.random.uniform(0.9,1.3)
                    else:
                        sample[0]=np.random.uniform(0.8,1.1)
                    starSample[i]=sample

                elif index==1:#M type star
                    if sample[1]<3250:
                        sample[0]=np.random.uniform(0.14,0.40)
                    else:
                        sample[0]=np.random.uniform(0.3,0.55)
                    starSample[i]=sample
                else:#K type star
                    if sample[1]>5000:
                        sample[0]=np.random.uniform(0.7,0.95)
                    else:
                        sample[0]=np.random.uniform(0.6,0.85)
                    starSample[i]=sample

        #Generate dependant parameters
        #Dependant parameters will be stored in a seperate list/dictionary
        #semi major axis, planet radius, planet mass, planet density, planet gravity, surface temperature, surface pressure,  pressure-temperature profile
        gStarDependant=[]
        mStarDependant=[]
        kStarDependant=[]
        for i,starSample in enumerate([gStarSamples,mStarSamples,kStarSamples]):
            for sample in starSample:
                starRad,starTemp=sample[0],sample[1]
                kappa,gamma1,gamma2,alpha=sample[2],sample[3],sample[4],sample[5]

                #Star luminostiy formula. 4πr^2*Temp^4*constant
                starLuminosity=4*np.pi*(starRad*const.R_sun.value)**2*starTemp**4*const.sigma_sb.value


                #Use starTemp to find the bounds on the semimajor axis
                #Approximation source: Kopparapu et al 2013, ApJ,
                #"Habitable Zones Around Main-sequence Stars: New Estimates"

                innerEdge= 1.7763 + 1.4335e-04 * (starTemp - 5780.) + 3.3954e-09 * (starTemp - 5780.)**2 - 7.6364e-12 * (starTemp - 5780.)**3 - 1.1950e-15 * (starTemp - 5780.)**4

                outerEdge=0.3207 + 5.4471e-05 * (starTemp - 5780.) + 1.5275e-09 * (starTemp - 5780.)**2 - 2.1709e-12 * (starTemp - 5780.)**3 - 3.8282e-16 * (starTemp - 5780.)**4
                
                semiMajorAxisMin = (starLuminosity / const.L_sun.value / innerEdge)**0.5
                semiMajorAxisMax = (starLuminosity / const.L_sun.value / outerEdge)**0.5

                # Generate a semimajor axis [AU]
                semiMajorAxis=np.random.uniform(semiMajorAxisMin, semiMajorAxisMax)

                #Generate planet parameters
                planetRad,planetMass,density,grav = generatePlanet(starLuminosity,semiMajorAxis)

                #Calculate surface pressures and temperature profile
                mean = 1.0
                stdv = 2.5
                lo   = 0.1
                hi   = 90.
                a, b = (lo - mean) / stdv, (hi - mean) / stdv
                rv   = ss.truncnorm(a, b, loc=mean, scale=stdv)
                surfPres = rv.rvs()
                pmin=1e-6#Bottom of thermosphere for Earth
                # Generate pressure array
                press=np.logspace(np.log10(pmin), np.log10(surfPres), num=50)

                #Generate temperature profile
                mean=0.95
                stdv=0.1
                lo=0.7
                hi=1.1
                a,b=(lo - mean) / stdv, (hi - mean) / stdv
                rv=ss.truncnorm(a, b, loc=mean, scale=stdv)
                beta= rv.rvs()
                #Pressure temperature profile
                PTprofile=generatePT(kappa, gamma1, gamma2, alpha, beta, pmin, surfPres, starRad, starTemp, semiMajorAxis, grav)
                # Extract surface temperature
                surfTemp = PTprofile[0,1]

                #semi major axis, planet radius, planet mass, planet density, planet gravity, surface temperature, surface pressure,  pressure-temperature profile
                dependantParameters=[semiMajorAxis,planetRad,planetMass,density,grav,surfTemp,surfPres, PTprofile]
                if i==0:#gStarSample
                    gStarDependant.append(dependantParameters)
                elif i==1:#mStarSample
                    mStarDependant.append(dependantParameters)
                else:#kStarSample
                    kStarDependant.append(dependantParameters)


        for i,starSample in enumerate([gStarSamples,mStarSamples,kStarSamples]):
            configs=[]#List of configs to give to PSG later
            for counter,sample in enumerate(starSample):
                configNum=(i*nSamples)+counter+1#+1 because counter starts at 0
                configFileName=os.path.join(configFolder,f"{atmosphereType}_{configNum}.txt")

                if i==0:
                    #G star
                    dependantParameters=gStarDependant[counter]
                    starType="G"
                elif i==1:
                    #M star
                    dependantParameters=mStarDependant[counter]
                    starType="M"
                else:
                    #K star
                    dependantParameters=kStarDependant[counter]
                    starType="K"
                    
                moleculeAbundances=calculateMoleculeAbundances()

                configs.append(configFileName)
                #Create config file
                createConfigFile(sample,dependantParameters,moleculeAbundances,starType,configFileName)


                if len(configs)==32:#Pass them in 32 chunks
                    callPSG(configs,atmosphereType)
                    configs=[]
                    # run=False
                    # break

               
            # if run==False:
            #     break
        

            if configs:
                callPSG(configs,atmosphereType)#Anything left over. Only for A1/A2 case, since 20000 is divisible by 32
        # if run==False:
            # break
                # print(moleculeAbundances)
                # break
            # break
        # break
    #220 seconds to do evrything before creating config file and runningto psg
    #Not as bad as I thought
    print(time.time()-start)








        
    # start=time.time()
    # a1=calculateMoleculeAbundances("A1")
    # a2=calculateMoleculeAbundances("A2")
    # b=calculateMoleculeAbundances("B")
    # c=calculateMoleculeAbundances("C")
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

    # print(f"Abundances for A1-Type: {a1}")

    # print(f"Abundances for A2-Type: {a2}")

    # print(f"Abundances for B-Type: {b}")

    # print(f"Abundances for C-Type: {c}")


    # print(time.time()-start)