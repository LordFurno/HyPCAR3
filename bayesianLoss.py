import numpy as np



#Will probably need to change this so it works with batches
#Will adress that later
def calculateLikelihood(yReal,ySim,sigma):
    '''
    This function calculates the likelhood P(A_pred|Y_real). It is a gaussian likelihood function.

    Inputs
    ------
    yReal: Real, input transmittance data
    ySim: Simulated transmittance data, calculated using the predicted abundances
    sigma: Uncertainty of how well predicted abundances translate to simulated spectra

    Returns
    -------
    likelihood: Likelihood value for given real and simulated data
    '''
    mse = np.mean((np.array(yReal) - np.array(ySim))**2)  # Mean squared error between real and simulated data

    return np.exp(-mse / (2 * sigma ** 2))

def classifyAtmosphere(predAbun):
    '''
    This function calculates the H, O, C and N values from the predicted abundances
    Then based on the mixing ratios, we will classify what type of atmosphere it is

    Inputs
    ------
    predAbun: The predicted abundances. A list of 7 values, representing the abundance of each molecule

    
    Returns
    -------
    atmosphereType: A value from the list [A1,A2,B,C]
    '''
    #The order of the molecules
    molecules={0:"O2",1:"N2",2:"H2",3:"CO2",4:"H2O",5:"CH4",6:"NH3"}

    #H = 2*H2 + 2*H2O + 3*NH3 + 4*CH4
    abundanceDict={}
    for i,val in enumerate(predAbun):
        abundanceDict[molecules[i]]=val

    H=2*abundanceDict["H2"] + 2*abundanceDict["H2O"] + 3*abundanceDict["NH3"] + 4*abundanceDict["CH4"]

    C=abundanceDict["CO2"] + abundanceDict["CH4"]

    O=2*abundanceDict["O2"] + 2*abundanceDict["CO2"] + abundanceDict["H2O"]

    N=2*abundanceDict["N2"] + abundanceDict["NH3"]

    if H> 2*O + 4*C:
        if 3*N < H - 2*O - 4*C:

            return "A1"
        else:
            return "A2"
        
    elif 2*O > H + 4*C:
        return "B"
    
    elif abs(H + C + O + N - 1) < 1e-3:  # Hydrogen-poor constraint
        return "C"

    else:
        return "Unkown"




def calculatePrior(predAbun):
    #First figure out what type of atmosphere the model think it is
    #Either A1, A2, B, C or unkown


    atmosphereType=classifyAtmosphere(predAbun)
    





def calculatePosterior(yReal,ySim,sigmaLikelihood,predAbun,muPrior,sigmaPrior):
    # Math: P({y_{real}}|A_{pred}) \propto P(A_{pred}|Y_{real}) * P(A_{pred})
    '''
    This function calculates the unnormalized posterior

    Inputs
    ------
    yReal: Real, input transmittance data
    ySim: Simulated transmittance data, calculated using the predicted abundances
    sigmaLikelihood: Uncertainty of how well predicted abundances translate to simulated spectra
    predAbun: Predicted abundances
    muPrior: Mean of the prior distrbution
    sigmaPrior: Uncertaintiy of the precicted abundances

    Returns
    -------
    posterior: Unnormalized posterior probability
    '''

    prior=
    # prior=calculatePrior(predAbun,muPrior,sigmaPrior)  
    likelihood=calculateLikelihood(yReal,ySim,sigmaLikelihood)
    #For the likelihood, I should just take the aggregated transmittance. However, later if I want to include wavelength-molecule mapping, here is where I would do it.
    #LIkelihood, I need both wavelength and transmittance
    # aggregated_likelihood=np.sum(likelihood,axis=0)

    # aggregated_likelihood=np.sum(np.log(likelihood), axis=0)
    # print(prior)
    # print(aggregated_likelihood)
    print(f"Likelihood: {likelihood**3}")
    print(f"Prior: {prior.tolist()}")
    # print(type(likelihood))
    posterior=likelihood**prior
    return posterior
