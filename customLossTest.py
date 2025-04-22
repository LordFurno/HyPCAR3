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

from physcicsFineTune.getAsync import get_data_async
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
        workingConfigFilePath=r"C:\Users\Tristan\Downloads\HyPCAR\workingDirectory\working-" +f"{index}.txt"
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
    return list(expected.values())

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
        if expectedValues==None:
            totalLoss+=0
        else:
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
    nll_mean = torch.mean(likelihood)


    #For the likelihood, I should just take the aggregated transmittance. However, later if I want to include wavelength-molecule mapping, here is where I would do it.

    # print(f"Likelihood: {nll_mean}")
    # print(f"Prior: {prior}")

    posterior=nll_mean+prior
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
        detectionTemp,predTemp=detectionOutput.cpu(),predAbun.cpu()

        
        hinge_present=F.relu(0.001 - predAbun)

        present_penalty=(detectionOutput*hinge_present).mean()

        #if detection says “absent”
        #punish pred_abun > 0.2
        #hinge_absent = max(0, pred_abun − 0.2)
        hinge_absent = F.relu(predAbun - 0.2)
        absent_penalty = ((1.0 - detectionOutput) * hinge_absent).mean()
        # 4) Combine
        detectionLoss = (80*present_penalty) + absent_penalty #This shou
        # predAbun=predAbun.to(device)

        dataLoss=self.dataBased(predAbun,realAbun)#Ranges from 0 to infinity


        
        
        sigmaLikelihood=0.1#How well simulated data matches predicted abundances. How accurate simulations are

        
        simulatedTransmittance=testNewAbundances(config,predAbun)
        simulatedTransmittance=simulatedTransmittance.to(device)


        #Need to figure out sigmaLikelihood and muPrior
        posterior=calculatePosterior(inputTransmittance,simulatedTransmittance,sigmaLikelihood,predAbun,uncertainty)
        
        #Combine the loss
        #For now, just add everything, but the in future, can play with the idea of multiplying each by some factor 
        with open("customLossTesting.txt","a") as f:
            val1,val2,val3=str(posterior.item()),str(dataLoss.item()),str(detectionLoss.item())
            f.write(val1+","+val2+","+val3+"\n")
        print((posterior,dataLoss,detectionLoss))
        return posterior+dataLoss+detectionLoss



molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]


random.seed(42)


testingData=[]
criterion=customLoss()

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
    model.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\pureCNN.pt",weights_only=True))
    model=model.to(device)


    detect=detect=detectionModel()
    detect.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\detectionModel.pt",weights_only=True))
    detect=detect.to(device)


    model.eval()
    detect.eval()
    runningLoss=0.0
    with torch.no_grad():


        total=0
        totalSamples=0
        counter=0
        for batch in tqdm(testingDataloader):
            data,labels,configs=batch

            data=data.to(device)
            # print(data[0])
            labels=labels.to(device)
            detectionOutput=detect(data)
        

            predAbun,uncertainties,attentionWeights=model(data,detectionOutput)

            predAbun=predAbun.to(device)
            uncertainties=uncertainties.to(device)
            detectionOutput=detectionOutput.to(device)

            loss=criterion(predAbun,uncertainties,labels,detectionOutput,configs,data)
            with open("customLossTesting.txt","a") as f:
                f.write(str(loss.item())+"\n")
            print(loss)
            runningLoss+=loss.item()
    print(runningLoss/len(testingDataloader))

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




