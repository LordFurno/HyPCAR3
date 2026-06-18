import torch
from torch.utils.data import DataLoader,Dataset,random_split
import pandas as pd
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from getAsync import get_data_async
import math
def oneHotEncoding(combination):
    '''
    This function will turn molecule combinations into a one-hot encoded vector

    Inputs
    ------
    combination: Tuple containing the abundances of the molecules in this order: "O2","N2","H2","CO2","H2O","CH4","NH3"

    Returns
    -------
    vector: One-hot encoded vector of 1's and 0's
    '''

    #The order of the molecules are: "O2","N2","H2","CO2","H2O","CH4","NH3"
    vector=[0.]*7
    for i,abundance in enumerate(combination):
        #At what point should a molecule be considered present? I don't know need to think about that
        if abundance>0.001:
            vector[i]=1.0
    return torch.tensor(vector) 

def kl_divergence(predictions, targets):
    '''
    This function calculates the Kullback-Leibler (KL) divergence between the predictions and the target value.

    Inputs
    ------
    predictions: A tensor that represents the models predictions
    targets: A tensor that represents the ground truth for the model

    Returns
    -------
    klDivergenceLoss: A value between 0 and 1 that represents the KL divergence loss. A value of 0 is best.
    '''
    predictions_log=torch.log(predictions+1e-10)  #Adding a small value to avoid log(0)
    kl_div=F.kl_div(predictions_log, targets, reduction='batchmean')
    return kl_div.item()


def topKAccuracy(output,target,k=1):
    '''
    This functions calculates the top-K accuracy for the predicted abundance values.
    
    Inputs
    ------
    output: The output from the abundance model
    target: The target abundance values
    k: k value for top-k accuracy.
    
    Returns
    -------
    Top-k value as a percentage of the values that match.
    '''
    _,topKPred = output.topk(k, dim=1)
    
    #Get the indices of the top k true abundances
    _,topKTarget = target.topk(k, dim=1)
    
    #Compare predictions with true targets
    correct = (topKPred == topKTarget).sum().item()
    
    #Return the accuracy as a percentage
    return correct/(target.size(0) * k)


def customCrossEntropy(output, target):
    '''
    This function calculate the cross-entropy loss between the predicted abundances and the true abundances.

    Inputs
    ------
    output: The output from the abundance model
    target: The target abundance values

    Returns:
    cross_entropy_loss: The average cross-entropy loss for the batch.
    '''

    #Apply log to predictions (log-softmax is typically used to stabilize computation)
    log_predictions=torch.log(output + 1e-9)  #Adding a small value to prevent log(0)
    
    #Element-wise multiplication of log_predictions with targets
    elementwise_loss=-target * log_predictions
    
    #Sum over the molecules (dim=1) to get the loss for each example in the batch
    cross_entropy_loss=torch.sum(elementwise_loss, dim=1)
    
    #Average over the batch
    cross_entropy_loss=torch.mean(cross_entropy_loss)
    
    return cross_entropy_loss

def getAbundances(fileName):
    '''
    This function grabs the molecule abundances from the config files and returns them as a vector

    Inputs
    ------
    fileName: The name of the config file.

    Returns
    -------
    abundances: A vector containing the abundance information
    '''
    abundances=[0.0]*7
    moleculeNames=["O2","N2","H2","CO2","H2O","CH4","NH3"]
    lines=[]
    with open(fileName) as f:
        for line in f:
            lines.append(line)

    abundances=lines[54]
    abundances=abundances.removeprefix("<ATMOSPHERE-LAYER-1>")
    abundances=abundances.split(",")

    if "None" in os.path.basename(fileName):
        #Special case
        abundances=list(map(float,abundances[2:9]))#Only gets target values, not background moolecules or 

    else:
        abundances=list(map(float,abundances[2:]))#Remove temperature profile information
    return abundances

def testNewAbundances(fileNames,moleculeAbundances):
    '''
    This function will update a temporary config file with the predictions from the ml model. 
    This is for the PSG-based loss, it takes in a batch of 32 different files and abundances. If this

    Inputs
    -----
    fileNames: The names of the config files
    moleculeAbundances: The new predicted abundances from the model.

    Returns
    -------
    combinedData: A tensor that contains the new wavelength and transmittance values for the updated abundances
    '''
    #Fix this functino, doesn't work because stupid batches. really really annoying, can think about using dask
    #To speed up the process because I'm dealing with about 32 files.
    #Super annoying, will create something called a "working directory" where I will store the modified config files per batch
    #Then use batch to run data through them, will need to pretty much rewrite everything here
    workingDirectory=[]
    for index,file in enumerate(fileNames):
        lines=[]
        with open(file) as f:
            for line in f:
                lines.append(line)
            
        workingConfigFilePath=os.path.join(os.environ["SLURM_TMPDIR"],"workingDirectory/")+f"working-{index}.txt"
        #Deal with atmosphere layers here
        #Start line is 54

        abundanceDictionary={}
        molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]
        for i in range(len(moleculeAbundances[index])):
            abundanceDictionary[molecules[i]]=moleculeAbundances[index][i]

        moleculeWeights={"O2":31.999, "N2":28.02, "H2":2.016,"CO2":44.01, "H2O":18.01528,"CH4":16.04,"NH3":17.03052 }#g/mol
        averageWeight=0
        for molecule in abundanceDictionary:
            averageWeight+=moleculeWeights[molecule]*abundanceDictionary[molecule]
        


        for i in range(50):
            atmosphereInfo=lines[54+i]
            atmosphereInfo=atmosphereInfo.removeprefix("<ATMOSPHERE-LAYER-"+str(i+1)+">")
            atmosphereInfo=atmosphereInfo.removesuffix("\n")
            atmosphereInfo=atmosphereInfo.split(",")

            atmosphereInfo[2:]=moleculeAbundances[index]
            
    

            lines[54+i]="<ATMOSPHERE-LAYER-"+str(i+1)+">"+",".join(map(str,list(atmosphereInfo)))+"\n"
        
        nMolecules=7

        HITRANValues={"O2":"HIT[7]","N2":"HIT[22]","H2":"HIT[45]","CO2":"HIT[2]","H2O":"HIT[1]","CH4":"HIT[6]","NH3":"HIT[11]"}

        #Additional parameters to actually run the data properly.
        lines[42]="<ATMOSPHERE-NGAS>"+str(len(moleculeAbundances[index]))+"\n" #Number of gases are in the atmosphere
        lines[43]="<ATMOSPHERE-GAS>"+",".join(molecules)+"\n" #What gases are in the atmosphere
        lines[44]="<ATMOSPHERE-TYPE>"+",".join(HITRANValues[mol] for mol in molecules)+"\n" #HITRAN values for each gas
        lines[45]="<ATMOSPHERE-ABUN>"+"1,"*(len(moleculeAbundances[index])-1)+"1"+"\n" #Molecule abunadnces. They're all 1, because abundances are defined in vertical profile
        lines[46]="<ATMOSPHERE-UNIT>"+"scl,"*(len(moleculeAbundances[index])-1)+"scl"+"\n" #Abundance unit
        lines[49]="<ATMOSPHERE-WEIGHT>"+str(averageWeight)+"\n" #Molecule weight of atmosphere g/mol
        lines[52]="<ATMOSPHERE-LAYERS-MOLECULES>"+",".join(molecules)+"\n" #Molecule in vertical profile

        with open(workingConfigFilePath,"w") as f:
            f.writelines(lines)
        workingDirectory.append(workingConfigFilePath)
    
    return get_data_async()#Calls the PSG docker to get the data


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
    
    mse = torch.mean((yReal - ySim)**2,dim=[1,2])  #Mean squared error between real and simulated data
    #return np.exp(-mse / (2 * sigma ** 2))
    nll = mse / (2 * sigma**2)
    #nll_aggregated = torch.mean(nll, dim=0)  #shape: (B,)
    #return nll_aggregated
    return nll


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
    
    elif abs(H + C + O + N - 1) < 1e-3:  #Hydrogen-poor constraint
        return "C"

    else:
        return "Unkown"


def calculateExpectedValues(predAbun):
    #First figure out what type of atmosphere the model think it is
    #Either A1, A2, B, C or unkown
    atmosphereType=classifyAtmosphere(predAbun)

    molecules={0:"O2",1:"N2",2:"H2",3:"CO2",4:"H2O",5:"CH4",6:"NH3"}
    abundanceDict={}
    for i,val in enumerate(predAbun):
        abundanceDict[molecules[i]]=val

    H=2*abundanceDict["H2"] + 2*abundanceDict["H2O"] + 3*abundanceDict["NH3"] + 4*abundanceDict["CH4"]

    C=abundanceDict["CO2"] + abundanceDict["CH4"]

    O=2*abundanceDict["O2"] + 2*abundanceDict["CO2"] + abundanceDict["H2O"]

    N=2*abundanceDict["N2"] + abundanceDict["NH3"]

    expected={}
    if atmosphereType=="A1":
        #For Type A1 atmospheres (H-rich, mainly H2O, CH4, NH3, and H2; lacking CO2, O2)
        '''
        H > 2O + 4C
        3N < H - 2O -4C

        D = H - N - 2C
        H2O = 2O / D
        NH3 = 2N / D
        CH4 = 2C / D
        H2 = (H - 2O - 4C - 3N) / D
        '''
        D = H - N - 2*C
        expected["H2O"]=2*O / D
        expected["NH3"]=2*N / D
        expected["CH4"]=2*C / D
        expected["H2"]=(H - 2*O - 4*C - 3*N) / D

        expected["O2"]=0
        expected["N2"]=0
        expected["CO2"]=0

    elif atmosphereType=="A2":
        #For Type A2 atmospheres (H-rich, but mainly H2O, CH4, NH3, and N2; lacking CO2, O2)
        '''
        D = H + 2C + 3N + 4O
        H2O = 6O / D
        NH3 = (2H - 8C - 4O) / D
        CH4 = 6C / D
        N2 = (3N + 4C + 2O - H) / D
        '''
        D = H + 2*C + 3*N + 4*O
        expected["H2O"]=6*O / D
        expected["NH3"]=(2*H - 8*C - 4*O) / D
        expected["CH4"]=6*C / D
        expected["N2"] =(3*N + 4*C + 2*O - H) / D

        expected["O2"]=0
        expected["H2"]=0
        expected["CO2"]=0

    elif atmosphereType=="B":
        #For Type B atmospheres (O-rich, mainly O2, N2, CO2, H2O; lacking NH3, H2)
        '''
        D = H + 2O + 2N
        H2O = 2H / D
        N2 = 2N / D
        CO2 = 4C / D
        O2 = (2O - H - 4C) /D
        '''
        D = H + 2*O + 2*N
        expected["H2O"]=2*H / D
        expected["N2"]=2*N / D
        expected["CO2"]=4*C / D
        expected["O2"]=(2*O - H - 4*C) / D

        expected["NH3"]=0
        expected["H2"]=0
        expected["CH4"]=0
    elif atmosphereType=="C":
        #For Type C atmospheres (Hydrogen-poor: mainly H2O, CO2, CH4, N2; lacking NH3, H2, O2)
        #Note: In Type C, the elemental budget is H + C + O + N = 1.
        '''
        Side conditions, so no negative results
        O > 0.5H + 2C -> O2-rich with no CH4
        H > 2O + 4C -> H2 ->H2-rich with no CO2
        C > 0.25H + 0.5O -> graphite condensation with no H2O

        H2O = (H + 2O - 4C) / (H + 2O + 2N)
        CH4 = (H - 2O + 4C) / (2H + 4O + 4N)
        CO2 = (2O + 4C - H) / (2H + 4O + 4N)
        N2 = 2N / (H + 2O + 2N)
        '''

        #Denom for H2O and N2
        D1 = H + 2*O + 2*N

        #Denom for CH4 and CO2
        D2 = 2*H + 4*O + 4*N  

        expected["H2O"] = (H + 2*O - 4*C) / D1
        expected["CH4"] = (H - 2*O + 4*C) / D2
        expected["CO2"] = (2*O + 4*C - H) / D2
        expected["N2"] = 2*N / D1

        expected["O2"] = 0
        expected["H2"] = 0
        expected["NH3"] = 0
    else:
        #Its unkown, prior should just be 1
        #So there is essentially no prior. Just want the likelihood
        return None
    return list(expected.values())
def calculatePrior(predAbun,sigmaPrior):
    '''
    predAbun: The predicted abundance for each moleecule
    sigmaPrior: The uncertainty for each molecule
    '''
    
    batchSize=predAbun.size(0)
    means=[]
    for i in range(batchSize):
        ev=calculateExpectedValues(predAbun[i].tolist())
        if ev is None:
            means.append(torch.zeros_like(predAbun[i]))
        else:
            means.append(torch.tensor(ev,device=predAbun.device))
    means=torch.stack(means,dim=0)


    normal = torch.distributions.Normal(0., 1.)
    #a = (0 - mu)/sigma,  b = (1 - mu)/sigma
    a = (0.0 - means) / sigmaPrior
    b = (1.0 - means) / sigmaPrior
    Z = normal.cdf(b) - normal.cdf(a)    #(B, M)

    #3) unnormalized quadratic term
    quad = (predAbun - means)**2 / (2 * sigmaPrior**2)


    nll_trunc = 0.5*torch.log(2*math.pi*sigmaPrior**2) + quad - torch.log(Z + 1e-12)  #add epsilon to avoid log(0)



    return nll_trunc.mean()
    



def calculatePosterior(yReal,ySim,sigmaLikelihood,predAbun,sigmaPrior):
    #Math: P({y_{real}}|A_{pred}) \propto P(A_{pred}|Y_{real}) * P(A_{pred})
    '''
    This function calculates the unnormalized posterior

    Inputs
    ------
    yReal: Real, input transmittance data
    ySim: Simulated transmittance data, calculated using the predicted abundances
    sigmaLikelihood: Uncertainty of how well predicted abundances translate to simulated spectra
    predAbun: Predicted abundances
    sigmaPrior: Uncertaintiy of the precicted abundances

    Returns
    -------
    posterior: Unnormalized posterior probability
    '''


    prior=calculatePrior(predAbun,sigmaPrior) 
    likelihood=calculateLikelihood(yReal,ySim,sigmaLikelihood)
    nll_mean = torch.mean(likelihood)

    #print(f"Likelihood: {nll_mean}")
    #print(f"Prior: {prior}")

    posterior=nll_mean+prior
    return posterior


class PSGSPSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, abundances, configs, eps=1e-3):
        """
        abundances: [batch, N_species]
        eps: small float (e.g. 1e-3)
        sim_args: any other constants your simulator needs
        """
        ctx.save_for_backward(abundances)
        ctx.eps = eps
        ctx.sim_args = configs

        #Run unperturbed simulation (we need this for the forward output)
        T0 = testNewAbundances(configs,abundances)
        T0=T0.to(device)
        return T0

    @staticmethod
    def backward(ctx, grad_T):
        abundances, = ctx.saved_tensors
        eps = ctx.eps
        sim_args = ctx.sim_args
        batch, N = abundances.shape

        #1) draw a random ±1 perturbation for each sample & species
        #   shape: [batch, N]
        delta = torch.randint(0, 2, (batch,N), device=abundances.device) * 2 - 1  
        
        #2) run two PSG calls at abund+εδ and abund−εδ
        T_plus  = testNewAbundances(sim_args,abundances + eps*delta)
        T_minus = testNewAbundances(sim_args,abundances - eps*delta)

        T_plus  = T_plus.to(device)
        T_minus = T_minus.to(device)

        #3) directional finite difference: shape [batch, n_wl]
        dT_dir = (T_plus - T_minus) / (2 * eps)
        #4) element-wise prod with grad_T, then sum over *both* spectral dims (1 & 2)
        #
        #chain rule: ∂L/∂a_j = ∑_λ (∂L/∂T_λ)·(∂T_λ/∂a_j).
        #   For SPSA: ∂T_λ/∂a_j ≈ dT_dir_λ * δ_j  (because 1/δ_j = δ_j when δ_j ∈ {±1})
        #   So grad_abund[:, j] = ∑_λ grad_T[:,λ] * dT_dir[:,λ] * δ[:,j]
        #   We can do this in one shot:
        #   - first compute batch-wise dot over λ: D = (grad_T * dT_dir).sum(dim=1)  [batch]
        #   - then multiply by δ for each species
        prod = grad_T * dT_dir                  #shape [B, 784, 2]
        D = prod.sum(dim=[1,2], keepdim=True)   #shape [B, 1, 1]
        D = D.view(batch, 1)                        #shape [B, 1]

        #5) broadcast across species to get [B, N_species]
        grad_abund = D * delta                  #shape [B, 7]
        #None for eps and sim_args (we don't backprop through those)
        
        return grad_abund, None, None
def detectionLoss(predAbun,detOutput):
    #if detection says “present”
    #punish pred_abun<0.001
    #hinge_present = max(0, 0.001 − predAbun)
    hinge_present=F.relu(0.001 - predAbun)
    present_penalty=(detOutput*hinge_present).mean()
    #if detection says “absent”
    #punish pred_abun > 0.2
    #hinge_absent = max(0, pred_abun − 0.2)
    hinge_absent = F.relu(predAbun - 0.2)
    absent_penalty = ((1.0 - detOutput) * hinge_absent).mean()
    detectionLoss = (80*present_penalty) + absent_penalty #This should make things work out better, and scale better

    return detectionLoss

class hypcarLoss(nn.Module):
    def __init__(self):
        super(hypcarLoss, self).__init__()
        self.dataBased=nn.MSELoss()
        self.dataLossLambda=nn.Parameter(torch.zeros(1))
        self.detectionLambda=nn.Parameter(torch.zeros(1))
        self.posteriorLambda=nn.Parameter(torch.zeros(1))
        self.log_sigma = nn.Parameter(torch.tensor(0.1))



    def forward(self,predAbun,uncertainty,realAbun,detectionOutput,config,inputTransmittance,batch):#Predicted abundances, actual abundances, config file, real data
        '''
        predAbun: Predicted abundances
        uncertainty: Uncertainty about predicted abundances
        realAbun: True abundances
        detectionOutput: Output from detection model
        config: Config file
        inputTransmittance: Input spectral data
        '''
        
        T_sim=PSGSPSAFunction.apply(predAbun,config,1e-3)

        #Detection loss
        #if detection says “present”
        #punish pred_abun<0.001
        #hinge_present = max(0, 0.001 − predAbun)
        hinge_present=F.relu(0.001 - predAbun)
        present_penalty=(detectionOutput*hinge_present).mean()
        #if detection says “absent”
        #punish pred_abun > 0.2
        #hinge_absent = max(0, pred_abun − 0.2)
        hinge_absent = F.relu(predAbun - 0.2)
        absent_penalty = ((1.0 - detectionOutput) * hinge_absent).mean()
        detectionLoss = (80*present_penalty) + absent_penalty #This should make things work out better, and scale better


        dataLoss=self.dataBased(predAbun,realAbun)#Ranges from 0 to infinity


        
        sigmaLikelihood=torch.exp(self.log_sigma)#How well simulated data matches predicted abundances. How accurate simulations are
        uncertainty=uncertainty.to(device)



        #Need to figure out sigmaLikelihood and muPrior
        posterior=calculatePosterior(inputTransmittance,T_sim,sigmaLikelihood,predAbun,uncertainty)
        posterior=posterior.to(device)
        


        #Combine the loss
        #See Kendall et al. “Multi‐Task Learning Using Uncertainty to Weigh Losses”

        totalLoss=(
            posterior*torch.exp(-self.posteriorLambda) + 0.5*self.posteriorLambda
            + dataLoss*torch.exp(-self.dataLossLambda) + 0.5*self.dataLossLambda
            + detectionLoss*torch.exp(-self.detectionLambda) + 0.5*self.detectionLambda


        )
        #For now, just add everything, but the in future, can play with the idea of multiplying each by some factor 
        return totalLoss
criterion=hypcarLoss()

outputs=torch.tensor([[0.6563708186149597, 0.33362287282943726, 2.8779223795738496e-11, 0.0014039903180673718, 0.008602352812886238, 5.618113974037442e-09, 3.2953798023704906e-10]])
uncertainties=torch.tensor([[0.07197261601686478, 0.6342238187789917, 0.3222094178199768, 0.4581497609615326, 0.4426732659339905, 1.5405185222625732, 1.438585877418518]])
detectionOutput=torch.tensor([[2.4381e-11, 9.0505e-01, 1.2884e-02, 9.6754e-01, 9.9762e-01, 9.9702e-01,1.0752e-03]])
labels=torch.tensor([[0.209,0.781,0.0,0.0003795,0.0035,0.0000017,0.0]])

dataLoss  = criterion.dataBased(outputs, labels)
detLoss   = detectionLoss(outputs, detectionOutput)
priorNLL  = calculatePrior(outputs,uncertainties)
loss      = dataLoss + detLoss + priorNLL
print(dataLoss)
print(detLoss)
print(priorNLL)
print(loss)
