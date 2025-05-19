import torch
from torch.utils.data import DataLoader,Dataset,random_split
import pandas as pd
import os
import numpy as np
import itertools
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score,multilabel_confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from scipy.stats import lognorm,norm,spearmanr, pearsonr

import matplotlib.pyplot as plt
import os

from getAsync import get_data_async
#Use module 3.1.6 for mpi4py


def wavelengthFilter(string):
    '''
    This function removes the um suffix from the wavelength data.

    Inputs
    ------
    string: A string to remove um from.

    Returns
    -------
    string: A float
    '''
    string=string.removesuffix(" um")
    return float(string)

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
    moleculeNames=["O2", "N2", "CO2", "H2O", "N2O", "CH4", "H2S"]
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

class customDataset(Dataset):
    def __init__(self,samples):#samples contain a listt of all file paths
        self.samples=samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        filePath=self.samples[index]


        configFilePath=r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles\\"
        # configFilePath="/home/tristanb/scratch/configFiles/"
        fileName=os.path.basename(filePath)
        fileName=fileName.removesuffix(".csv")
        configFilePath+=fileName+".txt"

        label=getAbundances(configFilePath)
        #Extract data from file
        data=pd.read_csv(filePath)
        wavelength=list(map(wavelengthFilter,data.iloc[:,0]))#Removes um from wavelength data
        transmittance=list(data.iloc[:,1])

        combinedData=torch.tensor(list(zip(wavelength, transmittance)), dtype=torch.float32)

        return combinedData,torch.tensor(label),configFilePath
    


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
        
        #C:\Users\Tristan\Downloads\HyPCAR\workingDirectory\working-0.txt
        workingConfigFilePath=r"C:\Users\Tristan\Downloads\HyPCAR3\workingDirectory\working-" +f"{index}.txt"
        #Deal with atmosphere layers here
        #Start line is 54

        abundanceDictionary={}
        molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]
        for i in range(len(moleculeAbundances[index])):
            abundanceDictionary[molecules[i]]=moleculeAbundances[index][i].cpu().item()

        moleculeWeights={"O2":31.999, "N2":28.02, "H2":2.016,"CO2":44.01, "H2O":18.01528,"CH4":16.04,"NH3":17.03052 }#g/mol
        averageWeight=0
        for molecule in abundanceDictionary:
            averageWeight+=moleculeWeights[molecule]*abundanceDictionary[molecule]
        


        for i in range(50):
            atmosphereInfo=lines[54+i]
            atmosphereInfo=atmosphereInfo.removeprefix("<ATMOSPHERE-LAYER-"+str(i+1)+">")
            atmosphereInfo=atmosphereInfo.removesuffix("\n")
            atmosphereInfo=atmosphereInfo.split(",")

            atmosphereInfo[2:]=moleculeAbundances[index].tolist()
            
    

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

class detectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv1d(in_channels=2, out_channels=128, kernel_size=5, stride=2)
        self.bn1=nn.BatchNorm1d(128)
        self.pool1=nn.MaxPool1d(2)

        self.conv2=nn.Conv1d(in_channels=128,out_channels=128,kernel_size=5,stride=2)
        self.bn2=nn.BatchNorm1d(128)
        self.pool2=nn.MaxPool1d(2)

        self.conv3=nn.Conv1d(in_channels=128,out_channels=64,kernel_size=3,stride=2)
        self.bn3=nn.BatchNorm1d(64)
        self.pool3=nn.MaxPool1d(2)

        self.conv4=nn.Conv1d(in_channels=64,out_channels=32,kernel_size=2,stride=2)
        self.bn4=nn.BatchNorm1d(32)
        self.pool4=nn.MaxPool1d(2)

        self.dropout1=nn.Dropout(0.4)

        self.global_pool = nn.AdaptiveAvgPool1d(1)



        self.fc1=nn.Linear(32,128)
        self.dropout2=nn.Dropout(0.75)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,7)#7 molecule present

    def forward(self,x):
        # Permute dimensions to [batch_size, channels, sequence_length]
        x=x.permute(0, 2, 1)
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.pool1(x)

        x=F.relu(self.bn2(self.conv2(x)))
        x=self.pool2(x)

        x=F.relu(self.bn3(self.conv3(x)))
        x=self.pool3(x)

        x=F.relu(self.bn4(self.conv4(x)))
        x=self.pool4(x)

        x=self.dropout1(x)

        x=self.global_pool(x)
        x=torch.flatten(x,1)

        x=F.relu(self.fc1(x))
        x=self.dropout2(x)

        x=F.relu(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))

        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()

        # Linear projections for Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into heads
        Q = Q.view(batch_size, seq_length, self.num_heads, input_dim // self.num_heads)
        K = K.view(batch_size, seq_length, self.num_heads, input_dim // self.num_heads)
        V = V.view(batch_size, seq_length, self.num_heads, input_dim // self.num_heads)

        # Transpose to (batch, heads, seq_len, feature_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (input_dim ** 0.5)
        attention_weights = self.softmax(scores)

        # Weighted sum of values
        weighted_sum = torch.matmul(attention_weights, V)

        # Concatenate heads and apply linear projection
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).contiguous()
        weighted_sum = weighted_sum.view(batch_size, seq_length, input_dim)

        # Output linear layer
        out = self.fc_out(weighted_sum)

        return out, attention_weights


class abundanceModel(nn.Module):#CHange to output uncertainty as well
    def __init__(self):
        super().__init__()

        self.fcDetect=nn.Linear(7, 64)


        self.conv1=nn.Conv1d(in_channels=2, out_channels=256, kernel_size=5, stride=2)
        self.bn1=nn.BatchNorm1d(256)
        self.pool1=nn.MaxPool1d(2)

        self.conv2=nn.Conv1d(in_channels=256,out_channels=512,kernel_size=7,stride=1)
        self.bn2=nn.BatchNorm1d(512)
        self.pool2=nn.MaxPool1d(2)

        self.conv3=nn.Conv1d(in_channels=512,out_channels=256,kernel_size=5,stride=2)
        self.bn3=nn.BatchNorm1d(256)
        self.pool3=nn.MaxPool1d(2)

        self.conv4=nn.Conv1d(in_channels=256,out_channels=64,kernel_size=2,stride=2)
        self.bn4=nn.BatchNorm1d(64)
        self.pool4=nn.MaxPool1d(2)

        self.dropout1=nn.Dropout(0.329397809173006)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.attention=MultiHeadAttention(input_dim=64, num_heads=16)


        self.fc_combined=nn.Linear(128, 128)#Combines both input branches (detection + data)


        self.flatten=nn.Flatten()


        self.dropout2=nn.Dropout(0.502219550897328)
        self.fc2=nn.Linear(128,64)

        self.fc3=nn.Linear(64,32)
        self.fc4=nn.Linear(32,7)#7 molecule present

        #Another branch for uncertaintiy values for each molecule
        self.fc_uncertainty=nn.Linear(32,7)

    def forward(self,x,detectionOutput):

        detectionOutput=F.relu(self.fcDetect(detectionOutput))


        #Permute dimensions to [batch_size, channels, sequence_length]
        x=x.permute(0, 2, 1)
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.pool1(x)

        x=F.relu(self.bn2(self.conv2(x)))
        x=self.pool2(x)

        x=F.relu(self.bn3(self.conv3(x)))
        x=self.pool3(x)

        x=F.relu(self.bn4(self.conv4(x)))
        x=self.pool4(x)


        x=self.dropout1(x)




        x=x.permute(0, 2, 1)
        x,attention_weights=self.attention(x)
        x=x.permute(0,2,1)
        x=self.global_pool(x)

        x=x.squeeze(-1)


        combined=torch.cat((x,detectionOutput), dim=1)
        combined=F.relu(self.fc_combined(combined))
        combined=F.relu(self.fc2(combined))
        combined=F.relu(self.fc3(combined))


        #Abundance branch
        logits=self.fc4(combined)
        #Apply softmax to make the output sum to 1 for abundances
        abundances=F.softmax(logits, dim=1)


        uncertaintyRaw=self.fc_uncertainty(combined)
        uncertainties=F.softplus(uncertaintyRaw)#Soft plus, since we don't want to bound to 1

        #In www.conf, found in etc php-fpm.d
        #listen = /tmp/run/php-fpm
        #And commented out listen.acl_gorups=apache,nginx
        #listen.moded=0666

        #Changed in etc/php-fpm.conf
        #Changed pid to equal /tmp/php-fpm.pid
        return abundances,uncertainties,attention_weights




def soft_prior_mean(predAbun, tau=1e-2):
    '''
    predAbun: [B, 7] tensor of abundances in order
        [O2, N2, H2, CO2, H2O, CH4, NH3]
    returns: [B, 7] tensor of smooth prior means μ
    '''
    B,M=predAbun.shape


    #elemental budgets
    O2, N2, H2, CO2, H2O, CH4, NH3 = predAbun.unbind(dim=1)
    H = 2*H2 + 2*H2O + 3*NH3 + 4*CH4
    C = CO2 + CH4
    O = 2*O2 + 2*CO2 + H2O
    N = 2*N2 + NH3

    #soft gates for A1/A2/B
    condA = H - (2*O + 4*C)             # >0 means A1/A2 region
    condA_gate = torch.sigmoid(condA / tau)   # sA in [0,1]

    condA2 = condA - (3*N - (H - 2*O - 4*C))  # >0 means A1, <=0 means A2
    gateA1 = condA_gate * torch.sigmoid( condA2 / tau)
    gateA2 = condA_gate * (1 - torch.sigmoid(condA2 / tau))

    condB = 2*O - (H + 4*C)
    gateB = (1 - condA_gate) * torch.sigmoid(condB / tau)

    #soft gate for C (hydrogen-poor)
    #we only enter C when not A or B
    #then we use |H+C+O+N - 1| smallness as indicator of HP
    condC_val = torch.abs(H + C + O + N - 1)
    gateC = (1 - condA_gate) * (1 - gateB) * torch.sigmoid(-condC_val / tau)

    #“unknown” residual (flat prior) if any left
    gateU = torch.clamp(1 - (gateA1 + gateA2 + gateB + gateC), min=0)

    #compute each regime’s expected μ vectors
    #A1:
    D_A1 = H - N - 2*C + 1e-6
    mu_A1 = torch.stack([
        torch.zeros_like(H),              # O2
        torch.zeros_like(H),              # N2
        (H - 2*O - 4*C - 3*N)/D_A1,       # H2
        torch.zeros_like(H),              # CO2
        2*O / D_A1,                       # H2O
        2*C / D_A1,                       # CH4
        2*N / D_A1                        # NH3
    ], dim=1)

    #A2:
    D_A2 = H + 2*C + 3*N + 4*O + 1e-6
    mu_A2 = torch.stack([
        torch.zeros_like(H),              # O2
        (3*N + 4*C + 2*O - H)/D_A2,       # N2
        torch.zeros_like(H),              # H2
        torch.zeros_like(H),              # CO2
        6*O / D_A2,                       # H2O
        6*C / D_A2,                       # CH4
        (2*H - 8*C - 4*O)/D_A2            # NH3
    ], dim=1)

    #B:
    D_B = H + 2*O + 2*N + 1e-6
    mu_B = torch.stack([
        (2*O - H - 4*C)/D_B,              # O2
        2*N / D_B,                        # N2
        torch.zeros_like(H),              # H2
        4*C / D_B,                        # CO2
        2*H / D_B,                        # H2O
        torch.zeros_like(H),              # CH4
        torch.zeros_like(H)               # NH3
    ], dim=1)

    #C: hydrogen-poor with side-conditions
    #we'll compute all four outputs, zeroing out those invalid by side-conditions
    D1 = H + 2*O + 2*N + 1e-6
    D2 = 2*H + 4*O + 4*N + 1e-6
    muH2O = (H + 2*O - 4*C) / D1
    muCH4 = (H - 2*O + 4*C) / D2
    muCO2 = (2*O + 4*C - H) / D2
    muN2  = 2*N / D1

    #enforce side-conditions softly
    #e.g. O2-rich if O > 0.5H + 2C => zero CH4
    cond_O2rich = torch.sigmoid((O - (0.5*H + 2*C)) / tau)
    cond_H2rich = torch.sigmoid((H - (2*O + 4*C)) / tau)
    cond_Cgraph = torch.sigmoid((C - (0.25*H + 0.5*O)) / tau)

    mu_C = torch.stack([
        torch.zeros_like(H),                            # O2
        muN2,                                           # N2
        torch.zeros_like(H),                            # H2
        muCO2 * (1 - cond_H2rich),                      # CO2
        muH2O * (1 - cond_Cgraph),                      # H2O
        muCH4 * (1 - cond_O2rich),                      # CH4
        torch.zeros_like(H)                             # NH3
    ], dim=1)
    # normalize C‐regime mix so sum to 1
    mu_C = mu_C / (mu_C.sum(dim=1, keepdim=True) + 1e-12)

    # 6) uniform fallback
    mu_U = torch.full_like(predAbun, 1.0 / M)

    # 7) final mixture
    gates = torch.stack([gateA1, gateA2, gateB, gateC, gateU], dim=1)  # [B,5]
    mus   = torch.stack([mu_A1, mu_A2, mu_B, mu_C, mu_U], dim=2)      # [B,7,5]

    mu = torch.sum(mus * gates.unsqueeze(1), dim=2)  # [B,7]
    return mu


def calculatePrior(predAbun, sigmaPrior):
    """
    Vectorized truncated-Gaussian prior using soft_prior_mean.
    Returns mean NLL over batch and species.
    """
    # 1) get smooth prior mean
    mu = soft_prior_mean(predAbun)  # [B,7]

    # 2) truncated normal cdf gap
    normal = torch.distributions.Normal(0., 1.)
    a = (0.0 - mu)           / sigmaPrior
    b = (1.0 - mu)           / sigmaPrior
    Z = normal.cdf(b) - normal.cdf(a)       # [B,7]
    Z = torch.clamp(Z, min=1e-6)            # avoid underflow

    # 3) quadratic term
    quad = (predAbun - mu)**2 / (2 * sigmaPrior**2)

    # 4) final NLL (drop constant log-term)
    nll_trunc = quad - torch.log(Z)

    # 5) mean over batch & species
    return nll_trunc.mean()



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
    # nll_aggregated = torch.mean(nll, dim=0)  # shape: (B,)
    # return nll_aggregated
    return nll



def calculatePosterior(yReal,ySim,sigmaLikelihood,predAbun,sigmaPrior):
    # Math: P({y_{real}}|A_{pred}) \propto P(A_{pred}|Y_{real}) * P(A_{pred})
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

    # print(f"Likelihood: {nll_mean}")
    # print(f"Prior: {prior}")

    posterior=nll_mean+prior
    return posterior
def project_onto_simplex(v):
    """
    Projects each row of v onto the probability simplex.
    v: [B, N]
    returns x: [B, N] with x >=0, sum(x)=1
    """
    B, N = v.shape
    # 1) sort v descending
    v_sorted, _ = torch.sort(v, descending=True, dim=1)       # [B, N]
    v_cumsum    = v_sorted.cumsum(dim=1)                     # [B, N]

    # 2) find rho
    js = torch.arange(1, N+1, device=v.device).view(1, -1)    # [1, N]
    rho_candidates = v_sorted + (1 - v_cumsum) / js           # [B, N]
    mask = rho_candidates > 0                                # [B, N]
    # argmax returns the first max index—since mask is True/False,
    # converting to int and taking argmax gives the last True
    rho = mask.to(torch.int).argmax(dim=1) + 1               # [B]

    # 3) compute theta
    idx   = rho - 1                                          # [B]
    theta = (1 - v_cumsum[torch.arange(B), idx]) / rho       # [B]

    # 4) project
    return torch.clamp(v + theta.view(B,1), min=0.0)         # [B, N]

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

        # Run unperturbed simulation (we need this for the forward output)
        T0 = testNewAbundances(configs,abundances)
        T0=T0.to(device)
        return T0

    @staticmethod
    def backward(ctx, grad_T):
        abundances, = ctx.saved_tensors
        eps = ctx.eps
        sim_args = ctx.sim_args
        batch, N = abundances.shape

        # 1) draw a random ±1 perturbation for each sample & species
        #    shape: [batch, N]
        delta = torch.randint(0, 2, (batch,N), device=abundances.device) * 2 - 1  
        

        A_plus=abundances + eps * delta
        A_minus=abundances - eps * delta

        a_plus  = project_onto_simplex(A_plus)
        a_minus = project_onto_simplex(A_minus)


        # 2) run two PSG calls at abund+εδ and abund−εδ


        T_plus  = testNewAbundances(sim_args,a_plus)
        T_minus = testNewAbundances(sim_args,a_minus)

        T_plus  = T_plus.to(device)
        T_minus = T_minus.to(device)

        # 3) directional finite difference: shape [batch, n_wl]
        dT_dir = (T_plus - T_minus) / (2 * eps)
        # 4) element-wise prod with grad_T, then sum over *both* spectral dims (1 & 2)
        #
        # chain rule: ∂L/∂a_j = ∑_λ (∂L/∂T_λ)·(∂T_λ/∂a_j).
        #    For SPSA: ∂T_λ/∂a_j ≈ dT_dir_λ * δ_j  (because 1/δ_j = δ_j when δ_j ∈ {±1})
        #    So grad_abund[:, j] = ∑_λ grad_T[:,λ] * dT_dir[:,λ] * δ[:,j]
        #    We can do this in one shot:
        #    - first compute batch-wise dot over λ: D = (grad_T * dT_dir).sum(dim=1)  [batch]
        #    - then multiply by δ for each species
        prod = grad_T * dT_dir                  # shape [B, 784, 2]
        D = prod.sum(dim=[1,2], keepdim=True)   # shape [B, 1, 1]
        D = D.view(batch, 1)                        # shape [B, 1]

        # 5) broadcast across species to get [B, N_species]
        grad_abund = D * delta                  # shape [B, 7]
        # None for eps and sim_args (we don't backprop through those)
        
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
    def __init__(self,
                 w_data: float  = 1.0,
                 w_det: float   = 20.0,
                 w_prior: float = 0.013,
                 w_like: float  = 0.2):
        super(hypcarLoss, self).__init__()
        self.dataBased=nn.MSELoss()
        self.log_sigma = nn.Parameter(torch.tensor(0.1))
        self.w_data  = w_data
        self.w_det   = w_det
        self.w_prior = w_prior
        self.w_like  = w_like


    def forward(self,predAbun,uncertainty,realAbun,detectionOutput,config,inputTransmittance):#Predicted abundances, actual abundances, config file, real data
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



        priorNLL=calculatePrior(predAbun, uncertainty)

        likNLL=calculateLikelihood(inputTransmittance,T_sim,sigmaLikelihood)
        likNLL=likNLL.mean()

        
        dataLoss*=self.w_data
        detectionLoss*=self.w_det
        priorNLL*=self.w_prior
        likNLL*=self.w_like


        # print(f"Predicted abundances: {predAbun}")
        # print(f"Real values: {realAbun}")
        # print(f"MSE loss: {dataLoss.item()}")
        # print(f"Detection loss: {detectionLoss.item()}")
        # print(f"Prior loss: {priorNLL}")
        # print(f"Likelihood loss: {likNLL}")


        return dataLoss+detectionLoss+priorNLL+likNLL



molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]


random.seed(42)


testingData=[]

baseMetrics = {
    "DataMSE":0.0,
    "DetectLoss":0.0,
    "PriorNLL":0.0,
    "LikNLL":0.0,
    "Count":0}
tunedMetrics = {
    "DataMSE":0.0,
    "DetectLoss":0.0,
    "PriorNLL":0.0,
    "LikNLL":0.0,
    "Count":0}


if __name__=="__main__":

    testSplit=0.1
    for atmosphereType in ["A","B","C"]:
        curFolderPath=r"C:\Users\Tristan\Downloads\HyPCAR3\data"
        curFolderPath+="\\"+atmosphereType
        files=[]
        for path in os.listdir(curFolderPath):
            #Need to get molecule abundances as well, this means that for each file, I need to go to the config file
            #Then extract the abundances there
            #This is how I will get the one-hot vector for the presence
            #Will just write a functionn

            files.append(os.path.join(curFolderPath,path))

        random.shuffle(files)

        testingSamples=[]
        for i,data in enumerate(files):
            path=data
            if i<(len(files)*testSplit):#Adds testing data
                testingSamples.append((path))
            else:
                break
        testingData.extend(testingSamples)

    print("DONE")


    random.shuffle(testingData)
    testingDataset=customDataset(testingData)
    testingDataloader=DataLoader(testingDataset,batch_size=32,shuffle=True)#Testing data loader

    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"


    model=abundanceModel()
    model.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\setWeightAbundance.pt",weights_only=True))
    model=model.to(device)

    base=abundanceModel()
    base.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\finalBaseAbundance.pt",weights_only=True))
    base=base.to(device)


    detect=detect=detectionModel()
    detect.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\flexibleDetectionModel.pt",weights_only=True))
    detect=detect.to(device)


    model.eval()
    detect.eval()
    runningLoss=0.0
    criterion=hypcarLoss()
    criterion=criterion.to(device)
    with torch.no_grad():
        physicsMse=0.0
        physicsMseBase=0.0
        fineTuneBetter=0
        totalLoss=0.0
        tunedLoss=0.0
#/home/tristanb/scratch/data
        total=0
        totalSamples=0
        counter=0
        for batch in tqdm(testingDataloader):
            data,labels,configs=batch

            data=data.to(device)
            labels=labels.to(device)
            detectionOutput=detect(data)
        
            #Tuned:
            predAbun,uncertainties,attentionWeights=model(data,detectionOutput)

            #Base:
            baseAbun,uncertainties2,attentionWeights2=base(data,detectionOutput)

            predAbun=predAbun.to(device)
            uncertainties=uncertainties.to(device)
            detectionOutput=detectionOutput.to(device)

            baseSim=PSGSPSAFunction.apply(baseAbun,configs, 1e-3).to(device)
            tunedSim=PSGSPSAFunction.apply(predAbun,configs, 1e-3).to(device)

            baseDataLoss=F.mse_loss(baseAbun,labels, reduction="mean")
            tunedDataLoss=F.mse_loss(predAbun,labels,reduction="mean")

            baseDetectionLoss=detectionLoss(baseAbun,detectionOutput)
            tunedDetectionLoss=detectionLoss(predAbun,detectionOutput)

            basePrior=calculatePrior(baseAbun,uncertainties2)
            tunedPrior=calculatePrior(predAbun,uncertainties)

            baseLik=calculateLikelihood(data,baseSim,0.1).mean()
            tunedLik=calculateLikelihood(data,tunedSim,0.1).mean()

            
            tunedMetrics["DataMSE"]+=tunedDataLoss.item()
            tunedMetrics["DetectLoss"]+=tunedDetectionLoss
            tunedMetrics["PriorNLL"]+=tunedPrior
            tunedMetrics["LikNLL"]+=tunedLik
            tunedMetrics["Count"]+=1

            baseMetrics["DataMSE"]+=baseDataLoss.item()
            baseMetrics["DetectLoss"]+=baseDetectionLoss
            baseMetrics["PriorNLL"]+=basePrior
            baseMetrics["LikNLL"]+=baseLik
            baseMetrics["Count"]+=1
            print(baseMetrics["PriorNLL"])
            print(tunedMetrics["PriorNLL"])

            # print(f"Fine-Tuneed: {phys_mse}")
            # print(f"Base: {phys_mseBase}")
            # if phys_mse<=phys_mseBase:
            #     fineTuneBetter+=1
            
            #Grab firstt value

            # realW,realT=[val[0] for val in data[0].tolist()],[val[1] for val in data[0].tolist()]
            # predW,predT=[val[0] for val in simData[0].tolist()],[val[1] for val in simData[0].tolist()]
            # baseW,baseT=[val[0] for val in simData2[0].tolist()],[val[1] for val in simData2[0].tolist()]
            # print(realW)
            # print(realT)
            # print("")
            # print(predW)
            # print(predT)
            # plt.plot(realW,realT,color="blue")
            # plt.plot(predW,predT,color="orange")
            # plt.plot(baseW,baseT,color="green")
            # plt.show(block=True)

            

            counter+=1
            if counter==100:
                break

#0.04435815446078777
# 0.039338372126221655
        print(baseMetrics)
        print(tunedMetrics)
        # print("Fine-tuned:")
        # print(physicsMse/100)
        # print("BAse:")
        # print(totalLoss/100)
        # print(tunedLoss/100)
        # # print(phys_mseBase/100)
        # print(fineTuneBetter)
            # print()
            # loss=criterion(predAbun,uncertainties,labels,detectionOutput,configs,data)
            # print(loss.item())
            

#0.12888 average physicsMSE
#For 4 batches
#4.12559
    #(tensor(3.9064), tensor(0.0138), tensor(0.2076, dtype=torch.float64))
    #(tensor(3.5244), tensor(0.0103), tensor(0.1830, dtype=torch.float64))
    #(tensor(4.4739), tensor(0.0141), tensor(0.2176, dtype=torch.float64))

#4.044241487979889
#3.9010595679283138
#4.056870102882385


            # if counter==0:
            #     # plt.figure(0)
            #     yReal=data[0]
            #     ySim=ySim[0]
            #     realX,realY=yReal[:,0],yReal[:,1]

            #     simX,simY=ySim[:,0],ySim[:,1]

            #     realX,realY=realX.cpu(),realY.cpu()
            #     simX,simY=simX.cpu(),simY.cpu()
                
            #     print()
            #     print(labels[0].tolist())
            #     print(predAbun[0].tolist())
            #     print(ySim)
            #     print(yReal)

            #     plt.plot(simX,simY,color="red",label="Simulated")
            #     plt.plot(realX,realY,color="blue",label="Real")


            #     plt.xlabel('X axis label')
            #     plt.ylabel('Y axis label')
            #     plt.title('Simulated vs Real Data')

            #     plt.show(block=True)

            # counter+=1

        
#295




