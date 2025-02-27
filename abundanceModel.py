import numpy as np

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

def calculatePrior(predAbun):
    #First figure out what type of atmosphere the model think it is
    #Either A1, A2, B or C
    predictionVector=[0.]*7
    for i,val in enumerate(predAbun):
        if val>0.001:
            predictionVector[i]=1.0




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
