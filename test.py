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
def getMoleculeCombinations(molecules):
    '''
    This functions take a list of molecules and then returns all possible combinations

    Inputs
    ------
    molecules: List of molecules

    Returns
    -------
    combinationList: List of combinations
    '''
    combinationList=[]
    for r in range(len(molecules)+1):
        combinationList.extend(itertools.combinations(molecules,r))
    return combinationList

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
    def __init__(self,samples):#samples contain a listt of all file paths
        self.samples=samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        filePath,label=self.samples[index]
        # /localscratch/tristanb.55643812.0/configFiles/A2_22622.txt
        # /localscratch/tristanb.55644187.0/configFiles/A2_22622.txt
        #Get config file for this sample
        configFilePath=os.path.join(os.environ["SLURM_TMPDIR"],"configFiles")
        # configFilePath="/home/tristanb/scratch/configFiles/"
        fileName=os.path.basename(filePath)
        fileName=fileName.removesuffix(".csv")
        configFilePath+=fileName+".txt"


        #Extract data from file
        data=pd.read_csv(filePath)
        wavelength=list(map(wavelengthFilter,data.iloc[:,0]))#Removes um from wavelength data
        transmittance=list(data.iloc[:,1])
        # if len(wavelength)!=784:
        #     print(filePath)
        # print(len(wavelength))
        combinedData=torch.tensor(list(zip(wavelength, transmittance)), dtype=torch.float32)
        # if torch.isnan(combinedData).any():
        #     print(filePath)
        return combinedData, label, configFilePath
        
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


def getLabel(filePath,specialMolecules=False):
    configFolder=os.path.join(os.environ["SLURM_TMPDIR"],"configFiles")
    # configFolder="/home/tristanb/scratch/configFiles"
    filePath=filePath.removesuffix(".csv")
    configFilePath=os.path.join(configFolder,filePath)
    configFilePath+=".txt"

    lines=[]
    with open(configFilePath) as f:
        for line in f:
            lines.append(line)


    abundances=lines[54]
    abundances=abundances.removepreifx("<ATMOSPHERE-LAYER-1>")
    abundances=abundances.split(",")
    if not specialMolecules:
        abundances=list(map(float,abundances[2:]))#Remove temperature profile information
        label=oneHotEncoding(abundances)
        return label
    else:
        abundances=list(map(float,abundances[2:9]))#Only gets target values, not background moolecules or 
        label=oneHotEncoding(abundances)
        return label



molecules=["O2", "N2", "CO2", "H2O", "N2O", "CH4", "H2S"]


random.seed(42)


testingData=[]


allSamples=[]
allLabels=[]
#

testSplit=0.15
for atmosphereType in ["A","B","C","None"]:
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
            label=getLabel(path,True)
        else:

            label=getLabel(path)
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
testingDataset=customDataset(testingData)
testingDataloader=DataLoader(testingDataset,batch_size=32,shuffle=True)#Testing data loader


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
numEpochs=15




f1_scores=[]
n_splits=5 #Number of folds (adjust as necessary)
cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


for trainIndex, valIndex in cv.split(allSamples, np.argmax(allLabels, axis=1)):#K-fold stratified cross validation

    xData=[]
    yData=[]
    
    for index in trainIndex:
        xData.append(allSamples[index])
    for index in valIndex:
        yData.append(allSamples[index])



    trainingDataset=customDataset(xData)
    validationDataset=customDataset(yData)
    
    trainingDataloader=DataLoader(trainingDataset,batch_size=32,shuffle=True)
    validationDataloader=DataLoader(validationDataset,batch_size=32,shuffle=True)

    model=detectionModel().to(device)
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.BCELoss()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #Model should work, but the data generation is a little weird, sometimess generates NAN values
    #After generation all the data might need to write a seperate script that goes through all data files and check for nan values
    #Then re runs the config file to get the actual data. Because the config files are never wrong.
    for epoch in range(numEpochs):
        model.train()
        running_loss=0.0
        correct=0
        total=0
        for batch in trainingDataloader:
            data,labels,config=batch

            data=data.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            outputs=model(data)

            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            #Calculate accuracy
            predicted=(outputs > 0.5).float()
            correct+=(predicted == labels).sum().item()
            total+=labels.numel()  #Total number of elements

        training_loss=running_loss / len(trainingDataloader)
        training_accuracy=100 * correct / total
        with open("HyPCAR_Detection_training.txt","a") as f:
            f.write(f"Epoch {epoch+1}, Training Loss: {training_loss}, Training Accuracy: {training_accuracy}%")
        print(f"Epoch {epoch+1}, Training Loss: {training_loss}, Training Accuracy: {training_accuracy}%")



        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            val_loss=0
            for batch in validationDataloader:
                data,labels,config=batch
                data=data.to(device)
                labels=labels.to(device)
                
                outputs=model(data)
                loss=criterion(outputs,labels)
                val_loss+=loss.item()

                predicted=(outputs > 0.5).float()
                correct+=(predicted == labels).sum().item()
                total+=labels.numel()  # Total number of elements
            valLoss=val_loss / len(validationDataloader)
            valAcc=100 * correct / total
            with open("HyPCAR_Detection_training.txt","a") as f:
                f.write(f"Epoch {epoch+1}, Validation Loss: {valLoss}, Validation Accuracy: {valAcc}%")
            print(f"Epoch {epoch+1}, Validation Loss: {valLoss}, Validation Accuracy: {valAcc}%")

torch.save(model.state_dict(), "detectionModel.pt")
model.eval()
test_loss=0
test_correct=0
test_total=0

all_predictions=[]
all_labels=[]

with torch.no_grad():
    for batch in testingDataloader:
        data,labels,config=batch

        data=data.to(device)
        labels=labels.to(device)

        
        outputs = model(data)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predicted=(outputs > 0.5).float()
        all_predictions.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        test_total+=labels.numel()
        test_correct+=(predicted == labels).float().sum().item()

test_loss/=len(testingDataloader)
test_accuracy=100 * test_correct / test_total

# Concatenate all predictions and labels
all_predictions=np.concatenate(all_predictions, axis=0)
all_labels=np.concatenate(all_labels, axis=0)

# Calculate F1 score
f1=f1_score(all_labels, all_predictions, average='macro')  # You can choose 'micro', 'macro', or 'weighted'
with open("HyPCAR_Detection_training.txt","a") as f:
    f.write(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%, Test F1 Score: {f1}")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%, Test F1 Score: {f1}")