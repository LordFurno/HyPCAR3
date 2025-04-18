
import torch
from torch.utils.data import DataLoader,Dataset,random_split
import pandas as pd
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

import time

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch

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


class abundanceModel(nn.Module):#Change to output uncertainty as well
    def __init__(self,config):
        super().__init__()

        self.fcDetect=nn.Linear(7, 64)


        self.conv1=nn.Conv1d(in_channels=2, 
                             out_channels=config["Conv1"], 
                             kernel_size=config["Kernel1"], 
                             stride=config["Stride1"],
                             padding=(config["Kernel1"]-1)//2)
        
        
        self.bn1=nn.BatchNorm1d(config["Conv1"])
        self.pool1=nn.MaxPool1d(2)

        self.conv2=nn.Conv1d(in_channels=config["Conv1"],
                             out_channels=config["Conv2"],
                             kernel_size=config["Kernel2"],
                             stride=config["Stride2"],
                             padding=(config["Kernel2"]-1)//2)
        
        
        self.bn2=nn.BatchNorm1d(config["Conv2"])
        self.pool2=nn.MaxPool1d(2)

        self.conv3=nn.Conv1d(in_channels=config["Conv2"],
                             out_channels=config["Conv3"],
                             kernel_size=config["Kernel3"],
                             stride=config["Stride3"],
                             padding=(config["Kernel3"]-1)//2)
        
        
        self.bn3=nn.BatchNorm1d(config["Conv3"])
        self.pool3=nn.MaxPool1d(2)

        self.conv4=nn.Conv1d(in_channels=config["Conv3"],
                             out_channels=config["Conv4"],
                             kernel_size=config["Kernel4"],
                             stride=config["Stride4"],
                             padding=(config["Kernel4"]-1)//2)
        
        
        self.bn4=nn.BatchNorm1d(config["Conv4"])
        self.pool4=nn.MaxPool1d(2)

        self.dropout1=nn.Dropout(config["Drop1"])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.attention=MultiHeadAttention(input_dim=32, num_heads=config["Heads"])


        self.fc_combined=nn.Linear(96, 128)#Combines both input branches (detection + data)
        self.flatten=nn.Flatten()
        self.dropout2=nn.Dropout(config["Drop2"])
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
    def __init__(self,samples):#samples contain a listt of all file paths
        self.samples=samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index][0],self.samples[index][1]
    

def loadData():
    molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]


    random.seed(42)


    testingData=[]


    allSamples=[]
    allLabels=[]

    testSplit=0.10
    for atmosphereType in ["A","B","C"]:
        dataPath="data/"+atmosphereType

        curFolderPath=os.path.join(os.environ["SLURM_TMPDIR"],dataPath)
        configFolderPath=os.path.join(os.environ["SLURM_TMPDIR"],"configFiles")

        files=[]
        for path in os.listdir(curFolderPath):

            fileName=path.removesuffix(".csv")
            configFilePath=os.path.join(configFolderPath,fileName)
            configFilePath+=".txt"

            label=getAbundances(configFilePath)
            files.append((os.path.join(curFolderPath,path),label))

        random.shuffle(files)

        testingSamples=[]
        for i,data in enumerate(files):
            path,label=data[0],data[1]
            if i<(len(files)*testSplit):#Adds testing data
                testingSamples.append((path,label))
            else:
                allSamples.append((path,label))
        testingData.extend(testingSamples)


    random.shuffle(testingData)





    preLoadedData=[]
    for dataPath,label in allSamples:#Pre-loads all the data, tqdm provides a progress bar so we can get an idea how long it will take
        
        configFilePath=os.path.join(os.environ["SLURM_TMPDIR"],"configFiles")

        fileName=os.path.basename(dataPath)
        fileName=fileName.removesuffix(".csv")
        configFilePath+=fileName+".txt"

        #Extract data from file
        data=pd.read_csv(dataPath)
        wavelength=list(map(wavelengthFilter,data.iloc[:,0]))#Removes um from wavelength data
        transmittance=list(data.iloc[:,1])
        combinedData=torch.tensor(list(zip(wavelength, transmittance)), dtype=torch.float32)

        preLoadedData.append((combinedData,torch.tensor(label),configFilePath))

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

    testingDataset=testDataset(testingData)
    testingDataloader=DataLoader(testingDataset,batch_size=32,shuffle=True)#Testing data loader

    return trainingDataloader,validationDataloader,testingDataloader

def evaluate(model,valLoader):
    model.eval()
    criterion=nn.MSELoss()


    kl=[]#KL-Divergence
    ce=[]#Cross-entropy
    topk=[]#Top-1 accuracy
    mse=[]#MSE
    with torch.no_grad():
        for i in range(3):#Number of validation
            runningMSE=0.0
            runnningKL=0.0  
            runningTopK=0.0    
            runningCE=0.0

            for batch in valLoader:
                data,labels,config=batch

                data=data.to(device)
                labels=labels.to(device)


                moleculeDetection=detectMolecules(data)

                outputs,uncertainty,attentionWeights=model(data,moleculeDetection)

                loss=criterion(outputs,labels)  

                runningMSE+=loss.item()
                runnningKL+=kl_divergence(outputs,labels)
                runningTopK+=topKAccuracy(outputs,labels,1)
                runningCE+=customCrossEntropy(outputs,labels)
            mse.append(runningMSE/len(valLoader))
            kl.append(runnningKL/len(valLoader))
            topk.append(runningTopK/len(valLoader))
            ce.append(runningCE/len(valLoader))
    mse=sum(mse)/len(mse)
    kl=sum(kl)/len(kl)
    topk=sum(topk)/len(topk)
    ce=sum(ce)/len(ce)

    return mse,kl,topk,ce

    

            



def objective(config,trainLoader,valLoader):#Train & evaluate the models
    try:
        model=abundanceModel(config).to(device)
    except:
        tune.report(mse=1e6, kl=1e6, topk=0.0, ce=1e6)#Really bad, since model is invalid, just incase seomthing breaks
        return

    optimizer=optim.Adam(model.parameters(),lr=config["lr"])
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    criterion=nn.MSELoss()
    numEpochs=10
    for epoch in range(numEpochs):
        model.train()
        for batch in trainLoader:
            data,labels,configs=batch

            data=data.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            detectionOutput=detectMolecules(data)

            outputs,uncertainty,attentionWeights=model(data,detectionOutput)

            loss=criterion(outputs,labels)

            loss.backward()

            optimizer.step()
    mseVal,klVal,topkVal,ceVal=evaluate(model,valLoader)

    tune.report(mse=mseVal, kl=klVal, topk=topkVal, ce=ceVal)


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




searchSpace={
    "Conv1": tune.choice([512, 256, 128, 64]),
    "Conv2": tune.choice([512, 256, 128, 64]),
    "Conv3": tune.choice([256, 128, 64, 32]),
    "Conv4": tune.choice([128,64,32]),

    "Kernel1": tune.choice([3,5,7]),
    "Kernel2": tune.choice([3,5,7]),
    "Kernel3": tune.choice([2,3,5]),
    "Kernel4": tune.choice([2,3]),

    "Stride1": tune.choice([1,2]),
    "Stride2": tune.choice([1,2]),
    "Stride3": tune.choice([1,2]),
    "Stride4": tune.choice([1,2]),

    "Drop1": tune.uniform(0.2,0.5),
    "Drop2": tune.uniform(0.5,0.9),

    "Heads": tune.choice([2, 4, 8, 16]),

    "lr": tune.loguniform(np.log(1e-5),np.log(1e-2))
}





'''
CNN model hyperparameters to optimize:
Output size for convolution layers:
    Conv1: 512, 256, 128, 64
    Conv2: 512, 256, 128, 64
    Conv3: 256, 128, 64, 32
    Conv4: 128,64,32

Kernel size for convolutional layers:
    Kernel1: 3, 5, 7
    Kernel2: 3, 5, 7
    Kernel3: 2, 3, 5
    Kernel4: 2, 3

Stride size for convolutional layers:
    Stride1: 1, 2
    Stride2: 1, 2
    Stride3: 1, 2
    Stride4: 1, 2
Dropout rates:
    Drop1: 0.2 - 0.5
    Drop2: 0.5 - 0.9

Num heads (Evenly divides input dim of 32):
    2, 4, 8, 16
Learning Rate:
    Log-scale:
    log (1e-5) to log(1e-2)

'''

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detect=detectionModel()
    detect=detect.to(device)

    #Connect to the Ray cluster launched in the job submission script
    ray.init(address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",_node_ip_address=os.environ['HEAD_NODE'])

    searchAlg = AxSearch(
    metric=["mse", "kl", "ce", "topk"],  
    mode=["min", "min", "min", "max"])

    trainingDataloader,validationDataloader,testingDataloader=loadData()

    analysis=tune.run(
        tune.with_parameters(objective, trainLoader=trainingDataloader, valLoader=validationDataloader),
        name="ray_tune_multi_objective",
        search_alg=searchAlg,
        num_samples=200,
        local_dir="/home/tristanb/projects/def-pjmann/tristanb/ray_results",
        resources_per_trial={"cpu": 4, "gpu": 1}
        )
    df=analysis.results_df
    analysis.export_report("report.html")
    df.to_csv("ray_tune_results.csv", index=False)


