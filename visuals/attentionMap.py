import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import os

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
    
def calculateChiSquared(yPred,yReal,sigma):
    chiElements=((yPred-yReal)**2)/(sigma**2)
    total=torch.sum(chiElements)
    totalPoints=yPred.numel()

    chiVal=total/totalPoints
    return chiVal.item()

molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]


random.seed(42)


testingData=[]


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

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=abundanceModel()
    model.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\pureCNN.pt",weights_only=True))
    model=model.to(device)


    detect=detect=detectionModel()
    detect.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\detectionModel.pt",weights_only=True))
    detect=detect.to(device)


    model.eval()
    detect.eval()
    with torch.no_grad():


        total=0
        totalSamples=0
        counter=0
        for batch in tqdm(testingDataloader):
            data,labels,configs=batch

            data=data.to(device)
            labels=labels.to(labels)
            detectionOutput=detect(data)
        

            predAbun,uncertainties,attentionWeights=model(data,detectionOutput)

            print(attentionWeights.size())

            attention_weights_np=attentionWeights.cpu().numpy()



            attention_weights_sample = attention_weights_np[0]#First sample in batch
            data_sample=data[0]#First data sample in batch
            


            seq_length = len(data_sample)
            upsampled_weights = torch.nn.functional.interpolate(
                attentionWeights, size=(seq_length, seq_length), mode="bilinear", align_corners=False
            )

            aggregated_attention = upsampled_weights[0].mean(dim=0).cpu().numpy()
            attention_sums = aggregated_attention.sum(axis=1)  
            

            print("REAL")
            for index,molecule in enumerate(molecules):
                print(f"{molecule}: {labels[0][index]*100}%")
            print()
            print("PRED")
            for index,molecule in enumerate(molecules):
                print(f"{molecule}: {predAbun[0][index]*100}%")
        

            print(F"Uncertainties: {uncertainties[0].tolist()}")
            print(f"Attention Weights: {attentionWeights[0].tolist()}")
            # print(un)
            wavelength = data_sample[:, 0].cpu().numpy()
            transmittance = data_sample[:, 1].cpu().numpy()
            #Aggregated attention scatterplot
            plt.figure(figsize=(10, 8))
            plt.scatter(wavelength, transmittance, c=attention_sums, cmap="viridis", s=100)
            plt.colorbar(label="Attention Intensity (Aggregated Across Heads)")
            plt.title("Attention on Wavelength vs. Transmittance (Aggregated Heads)")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Transmittance")
            plt.show()
            #Real Abundances: [0.0, 0.0, 0.032601602375507355, 0.0, 0.20410604774951935, 0.1226552352309227, 0.6406370997428894]
            #Predicted Abundances: [7.996823114808649e-06, 0.09490162879228592, 0.14875809848308563, 2.9451175578287803e-06, 0.16910181939601898, 0.07918565720319748, 0.5080417990684509]
            #Uncertainties: [0.9459805488586426, 0.7382314801216125, 0.8511866331100464, 0.464542955160141, 0.8512697815895081, 0.38728490471839905, 0.5241609811782837]
            '''
            #This code does the scatter plott per head
            attention_head = upsampled_weights[0, 0].detach().numpy()

            # Sum attention values for each wavelength-transmittance pair
            attention_sums = attention_head.sum(axis=1)

            # Plot the scatterplot
            plt.figure(figsize=(10, 8))
            plt.scatter(wavelength, transmittance, c=attention_sums, cmap="viridis", s=100)
            plt.colorbar(label="Attention Intensity")
            plt.title("Attention on Wavelength vs. Transmittance")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Transmittance")
            plt.show()
            '''



            #Heatmap for attention weights. 
            num_heads = attention_weights_sample.shape[0]
            plt.figure(figsize=(10, 5))  # Adjust figure size based on the number of heads
            for i in range(num_heads):
                ax = plt.subplot(2, 4, i + 1)  # Create a 2x4 grid for 8 attention heads
                sns.heatmap(attention_weights_sample[i], annot=True, cmap='viridis', ax=ax)
                ax.set_title(f'Head {i+1}')
                ax.set_xticks([])  # Hide x and y ticks for clarity
                ax.set_yticks([])




            plt.tight_layout()
            plt.show()
            counter+=1




