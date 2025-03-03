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
from scipy.stats import lognorm,norm,spearmanr, pearsonr
from sklearn.model_selection import StratifiedKFold


import matplotlib.pyplot as plt
import time
from runPsg import get_data_async


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
    
    # Get the indices of the top k true abundances
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

    # Apply log to predictions (log-softmax is typically used to stabilize computation)
    log_predictions=torch.log(output + 1e-9)  # Adding a small value to prevent log(0)
    
    # Element-wise multiplication of log_predictions with targets
    elementwise_loss=-target * log_predictions
    
    # Sum over the molecules (dim=1) to get the loss for each example in the batch
    cross_entropy_loss=torch.sum(elementwise_loss, dim=1)
    
    # Average over the batch
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
            
        workingConfigFilePath="/home/tristanb/projects/def-pjmann/tristanb/workingDirectory/working-"+f"{index}.txt"
        #Deal with atmosphere layers here
        #Start line is 54

        abundanceDictionary={}
        molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]
        for i in range(len(moleculeAbundances)):
            abundanceDictionary[molecules[i]]=moleculeAbundances[i]

        moleculeWeights={"O2":31.999, "N2":28.02, "H2":2.016,"CO2":44.01, "H2O":18.01528,"CH4":16.04,"NH3":17.03052 }#g/mol
        averageWeight=0
        for molecule in abundanceDictionary:
            averageWeight+=moleculeWeights[molecule]*abundanceDictionary[molecule]
        


        for i in range(50):
            atmosphereInfo=lines[54+i]
            atmosphereInfo=atmosphereInfo.removeprefix("<ATMOSPHERE-LAYER-"+str(i+1)+">")
            atmosphereInfo=atmosphereInfo.removesuffix("\n")
            atmosphereInfo=atmosphereInfo.split(",")

            atmosphereInfo[2:]=moleculeAbundances
            
    

            lines[54+i]="<ATMOSPHERE-LAYER-"+str(i+1)+">"+",".join(map(str,list(atmosphereInfo)))+"\n"
        
        nMolecules=7

        HITRANValues={"O2":"HIT[7]","N2":"HIT[22]","H2":"HIT[45]","CO2":"HIT[2]","H2O":"HIT[1]","CH4":"HIT[6]","NH3":"HIT[11]"}

        #Additional parameters to actually run the data properly.
        lines[42]="<ATMOSPHERE-NGAS>"+str(len(moleculeAbundances))+"\n" #Number of gases are in the atmosphere
        lines[43]="<ATMOSPHERE-GAS>"+",".join(molecules)+"\n" #What gases are in the atmosphere
        lines[44]="<ATMOSPHERE-TYPE>"+",".join(HITRANValues[mol] for mol in molecules)+"\n" #HITRAN values for each gas
        lines[45]="<ATMOSPHERE-ABUN>"+"1,"*(len(moleculeAbundances)-1)+"1"+"\n" #Molecule abunadnces. They're all 1, because abundances are defined in vertical profile
        lines[46]="<ATMOSPHERE-UNIT>"+"scl,"*(len(moleculeAbundances)-1)+"scl"+"\n" #Abundance unit
        lines[49]="<ATMOSPHERE-WEIGHT>"+str(averageWeight)+"\n" #Molecule weight of atmosphere g/mol
        lines[52]="<ATMOSPHERE-LAYERS-MOLECULES>"+",".join(molecules)+"\n" #Molecule in vertical profile

        with open(workingConfigFilePath,"w") as f:
            f.writelines(lines)
        workingDirectory.append(workingConfigFilePath)
    
    return get_data_async()#Calls the PSG docker to get the data


def detectMolecules(data):
    '''
    This function will pass the data through the detection model and return its raw output

    Inputs
    ------
    data: A tensor of batch 32 that contains the wavelength, transmittance data
    
    
    Returns
    -------
    outputs: A vector that represents the detection models output
    '''
    #Load the saved model weights
    
    detect.load_state_dict(torch.load("/home/tristanb/projects/def-pjmann/tristanb/detectionModel.pt",weights_only=True))
    with torch.no_grad():
        outputs=detect(data)

    return outputs

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
 


class customDataset(Dataset):
    def __init__(self,samples):
        self.samples=samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index][0],self.samples[index][1],self.samples[index][2]

class testDataset(Dataset):
    def __init__(self,samples):#samples contain a list of all file paths
        self.samples=samples

        self.data=[]
        self.labels=[]
        self.configs=[]
        for filePath in samples:

            configFilePath=os.path.join(os.environ["SLURM_TMPDIR"],"configFiles")

            fileName=os.path.basename(filePath)
            fileName=fileName.removesuffix(".csv")
            configFilePath+=fileName+".txt"

            label=getAbundances(configFilePath)#Gets molecule abundances from config file
            #Extract data from file
            data=pd.read_csv(filePath)
            wavelength=list(map(wavelengthFilter,data.iloc[:,0]))#Removes um from wavelength data
            transmittance=list(data.iloc[:,1])
       

            combinedData=torch.tensor(list(zip(wavelength, transmittance)), dtype=torch.float32)

        
            self.data.append(combinedData)
            self.labels.append(torch.tensor(label))
            self.configs.append(configFilePath)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        return self.data[index], self.labels[index], self.configs[index]
    
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

        self.flatten=nn.Flatten()

        self.fc1=nn.Linear(64,128)
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

        x=torch.flatten(x, 1)

        x=F.relu(self.fc1(x))
        x=self.dropout2(x)
        
        x=F.relu(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))
        
        return x
    
class abundanceModel(nn.Module):#CHange to output uncertainty as well
    def __init__(self):
        super().__init__()

        self.fcDetect=nn.Linear(7, 64)#7 molecules to be detected

        
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


        self.attention=MultiHeadAttention(input_dim=32, num_heads=8)


        self.fc_combined=nn.Linear(128, 128)#Combines both input branches (detection + data)


        self.flatten=nn.Flatten()


        self.dropout2=nn.Dropout(0.75)
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

        x, attention_weights=self.attention(x.permute(0, 2, 1))  #Switch back to (batch, seq_len, feature_dim)


        x=torch.flatten(x, 1)

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

    mse = torch.mean((yReal - ySim)**2,dim=1)  #Mean squared error between real and simulated data
    #return np.exp(-mse / (2 * sigma ** 2))
    nll = mse / (2 * sigma**2)

    # Optionally include constant term: nll += 0.5 * np.log(2 * np.pi * sigma**2)
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
    
    elif abs(H + C + O + N - 1) < 1e-3:  # Hydrogen-poor constraint
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
    return expected.values()

def calculatePrior(predAbun,sigmaPrior):
    '''
    predAbun: The predicted abundance for each moleecule
    sigmaPrior: The uncertainty for each molecule
    '''
    totalLoss=0.0
    batchSize=predAbun.shape[0]
    for i in range(batchSize):
        sampleLoss=0.0
        sampleAbundance=predAbun[i]
        sampleSigma=sigmaPrior[i]
        expectedValues=calculateExpectedValues(sampleAbundance.tolist())
        for j in range(len(sampleAbundance)):
            mu=expectedValues[j]
            sigma=sampleSigma[j]

            y=sampleAbundance[j]
            sampleLoss+=np.log(np.sqrt(2 * np.pi) * sigma) + ((y - mu)**2) / (2 * sigma**2)
        totalLoss+=sampleLoss
    return totalLoss/batchSize



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


    #For the likelihood, I should just take the aggregated transmittance. However, later if I want to include wavelength-molecule mapping, here is where I would do it.

    print(f"Likelihood: {likelihood}")
    print(f"Prior: {prior}")

    posterior=likelihood+prior
    return posterior


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.dataBased=nn.MSELoss()

    def forward(self,predAbun,uncertainty,realAbun,detectionOutput,config,inputTransmittance):#Predicted abundances, actual abundances, config file, real data
        '''
        predAbun: Predicted abundances
        uncertainty: Uncertainty about predicted abundances
        realAbun: True abundances
        detectionOutput: Output from detection model
        config: Config file
        inputTransmittance: Input spectral data
        '''
        
        #How should I compare detection output to abundance output. Hmm, could be ranked based, 
        #Like molecules with more confidence in exsisting should have higher abundance

        spearman,pValue=spearmanr(detectionOutput,predAbun)
        detectionLoss=1.0-spearman#Best case is 0, worst case is 1. Boundeed between 1 and 0


        dataLoss=self.dataBased(predAbun,realAbun)#Ranges from 0 to infinity


        

        sigmaLikelihood=0.1#How well simulated data matches predicted abundances. How accurate simulations are

        
        simulatedTransmittance=testNewAbundances(config,predAbun)
        simulatedTransmittance=simulatedTransmittance.to(device)


        #Need to figure out sigmaLikelihood and muPrior
        posterior=calculatePosterior(inputTransmittance,simulatedTransmittance,sigmaLikelihood,predAbun,uncertainty)
        
        #Combine the loss
        #For now, just add everything, but the in future, can play with the idea of multiplying each by some factor 
        return posterior+dataLoss+detectionLoss








if __name__ == '__main__':
    molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]


    random.seed(42)


    testingData=[]


    allSamples=[]
    allLabels=[]


    testSplit=0.10

    for atmosphereType in ["A","B","C"]:
        dataPath="data/"+atmosphereType
        curFolderPath=os.path.join(os.environ["SLURM_TMPDIR"],dataPath)
        files=[]
        for path in os.listdir(curFolderPath):
            #Need to get molecule abundances as well, this means that for each file, I need to go to the config file
            #Then extract the abundances there
            #This is how I will get the one-hot vector for the presence
            #Will just write a functionn
            if atmosphereType=="None":
                #Gett labels in a special way
                label=getAbundances(path)
            else:

                label=getAbundances(path)
            files.append((os.path.join(curFolderPath,path),label))

        random.shuffle(files)

        testingSamples=[]
        for i,data in enumerate(files):
            path,label=data[0],data[1]
            if i<(len(files)*testSplit):#Adds testing data
                testingSamples.append((path,label))
            else:
                allSamples.append((path,label))
                allLabels.append(label)
        testingData.extend(testingSamples)


    random.shuffle(testingData)



    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detect=detectionModel()
    detect=detect.to(device)

    print(device)


    numEpochs=10



    '''
    Consider pre-loading the data set before training even beings. This would make it so that I only ever have to load
    the data once. Then the batches will be super fast, even for a seperate validation split.
    '''

    start1=time.time()
    preLoadedData=[]
    for sample in allSamples:#Pre-loads all the data, tqdm provides a progress bar so we can get an idea how long it will take
        
        configFilePath=os.path.join(os.environ["SLURM_TMPDIR"],"configFiles")

        fileName=os.path.basename(sample)
        fileName=fileName.removesuffix(".csv")
        configFilePath+=fileName+".txt"

        label=getAbundances(configFilePath)#Gets molecule abundances from config file
        #Extract data from file
        data=pd.read_csv(sample)
        wavelength=list(map(wavelengthFilter,data.iloc[:,0]))#Removes um from wavelength data
        transmittance=list(data.iloc[:,1])
        combinedData=torch.tensor(list(zip(wavelength, transmittance)), dtype=torch.float32)

        preLoadedData.append((combinedData,torch.tensor(label),configFilePath))

    print(f"Loaded all training data, it took: {time.time()-start1}")
    random.shuffle(preLoadedData)


    nTotal=len(preLoadedData)
    nTrain=int(0.8*nTotal)

    trainingData=preLoadedData[:nTrain]
    validationData=preLoadedData[nTrain:]

    #Preloaded data is: data, label, configFile
    trainingDataset=customDataset(trainingData)
    validationDataset=customDataset(validationData)

    trainingDataloader=DataLoader(trainingDataset,batch_size=32,shuffle=True)
    validationDataloader=DataLoader(validationDataset,batch_size=32,shuffle=True)
    


    model=abundanceModel().to(device)
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)



    criterion=customLoss()

    for epoch in range(numEpochs):
        model.train()
        running_loss = 0.0
        running_KL_loss = 0.0  
        running_top_k = 0.0    
        running_cross_entropy = 0.0  

        for batch in trainingDataloader:
            data,labels,configs=batch

            data=data.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            detectionOutput=detectMolecules(data)

            outputs,uncertainties,attentionWeights=model(data,detectionOutput)


            loss=criterion(outputs,uncertainties,labels,detectionOutput,config,data)

            
            # print(outputs.size())
            # print(labels.size())
            # print("")

            running_loss+=loss.item()

            running_KL_loss+=kl_divergence(outputs,labels)
            running_top_k+=topKAccuracy(outputs,labels,1)
            running_cross_entropy+=customCrossEntropy(outputs,labels)
            
            
            loss.backward()
            optimizer.step()
    
            


        # print(time.time()-start)

        trainingLoss=running_loss/len(trainingDataloader)
        klLoss=running_KL_loss/len(trainingDataloader)
        topKAcc=running_top_k/len(trainingDataloader)
        crossEntropy=running_cross_entropy/len(trainingDataloader)


        #Instead of weighted accuracy, use top-k accuracy and R^2 value. 
        with open("HyPCAR_Abundance_Training.txt",'a') as f:
            f.write(f"Epoch {epoch+1}, Loss: {trainingLoss}, KL Divergence: {klLoss}, Top-K Accuracy: {topKAcc}, Cross Entropy: {crossEntropy}"+"\n")

        print(f"Epoch {epoch+1}, Loss: {trainingLoss}, KL Divergence: {klLoss}, Top-K Accuracy: {topKAcc}, Cross Entropy: {crossEntropy}")


        model.eval()
        with torch.no_grad():
            val_loss=0
            validation_KL_loss=0.0
            
            validation_top_k=0.0
            validation_cross_entropy=0.0

            counter=0
            for batch in validationDataloader:
                data,labels,config=batch

                data=data.to(device)
                labels=labels.to(device)

                optimizer.zero_grad()

                moleculeDetection=detectMolecules(data)

                outputs,attentionWeights=model(data,moleculeDetection)

                loss=criterion(outputs,uncertainties,labels,detectionOutput,config,data)

                val_loss+=loss.item()
                validation_KL_loss+=kl_divergence(outputs,labels)
                validation_top_k+=topKAccuracy(outputs,labels,1)
                validation_cross_entropy+=customCrossEntropy(outputs,labels)

                

            valLoss=val_loss/len(validationDataloader)
            valKL=validation_KL_loss/len(validationDataloader)
            valTopK=validation_top_k/len(validationDataloader)
            valCrossEntropy=validation_cross_entropy/len(validationDataloader)
            with open("HyPCAR_Abundance_Training.txt",'a') as f:
                f.write(f"Validation Loss: {valLoss}, KL Divergence: {valKL}, Top-K Accuracy: {valTopK}, Cross Entropy: {valCrossEntropy}"+"\n")
            print(f"Validation Loss: {valLoss}, KL Divergence: {valKL}, Top-K Accuracy: {valTopK}, Cross Entropy: {valCrossEntropy}")

    torch.save(model.state_dict(), "HyPCAR_Abundance.pt")
    model.eval()

    testingDataset=testDataset(testingData)
    testingDataloader=DataLoader(testingDataset,batch_size=32,shuffle=True)#Testing data loader

    test_loss=0
    regularLoss=nn.MSELoss()
    test_kl_loss=0.0
    test_top_k=0.0
    test_cross_entropy=0.0
    with torch.no_grad():
        for batch in testingDataloader:
            data,labels,config=batch

            data=data.to(device)
            labels=labels.to(device)

            
            moleculeDetection=detectMolecules(data)
            
            outputs,attentionWeights=model(data,moleculeDetection)


            loss=regularLoss(outputs,labels)


            test_loss+=loss.item()
            test_kl_loss+=kl_divergence(outputs,labels)
            test_top_k+=topKAccuracy(outputs,labels,1)
            test_cross_entropy+=customCrossEntropy(outputs,labels)
            

        test_loss=test_loss/len(testingDataloader)
        testKL=test_kl_loss/len(testingDataloader)
        testTopK=test_top_k/len(testingDataloader)
        testCrossEntropy=test_cross_entropy/len(testingDataloader)

    with open("HyPCAR_Abundance_Training.txt",'a') as f:
        f.write(f"Testing Loss: {test_loss}, KL Divergence: {testKL}, Top-K Accuracy: {testTopK}, Cross Entropy: {testCrossEntropy}"+"\n")
    # print(f"Testing Loss: {test_loss}, KL Divergence: {testKL}, Top-K Accuracy: {testTopK}, Cross Entropy: {testCrossEntropy}")
