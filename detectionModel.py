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
    combination: Tuple that contains all molecules present

    Returns
    -------
    vector: One-hot encoded vector of 1's and 0's
    '''

    #The order of the molecules are: "O2", "N2", "CO2", "H2O", "N2O", "CH4", "H2S"
    vector=[0.]*7
    moleculeIndexes={"O2":0, "N2":1, "CO2":2, "H2O":3, "N2O":4, "CH4":5, "H2S":6}
    for molecule in combination:
        vector[moleculeIndexes[molecule]]=1.0
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

        #Get config file for this sample
        configFilePath=r"C:\Users\Tristan\Downloads\HyPCAR\configFiles"+f"\\"
        fileName=os.path.basename(filePath)
        fileName=fileName.removesuffix(".csv")
        configFilePath+=fileName+".txt"

        #Encode label
        label=oneHotEncoding(label)
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

molecules=["O2", "N2", "CO2", "H2O", "N2O", "CH4", "H2S"]
folderPath=r"C:\Users\Tristan\Downloads\HyPCAR\data"
moleculeCombinations=getMoleculeCombinations(molecules)
random.seed(42)


testingData=[]


allSamples=[]
allLabels=[]


testSplit=0.15

for combination in moleculeCombinations:
    if combination==():
        curFolderPath=r"C:\Users\Tristan\Downloads\HyPCAR\data\None"
    else:
        curFolderPath=folderPath+f"\\{"-".join(combination)}"

    files=[]#Shuffles all files in molecule combination directory
    for path in os.listdir(curFolderPath):
        #I'll just add molecule label here to make it easier
        files.append((os.path.join(curFolderPath,path),combination))
    random.shuffle(files)


    testingSamples=[]#Contains testing data for that molecule combination
    for i,data in enumerate(files):
        path,combination=data[0],data[1]
        if i<(len(files)*testSplit):#Adds testing data
            testingSamples.append((os.path.join(curFolderPath,path),combination))
        else:#Adds rest of data to training/validation
            allSamples.append((os.path.join(curFolderPath,path),combination))
            allLabels.append(oneHotEncoding(combination))
    testingData.extend(testingSamples)


random.shuffle(testingData)


testingDataset=customDataset(testingData)
testingDataloader=DataLoader(testingDataset,batch_size=32,shuffle=True)#Testing data loader


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

numEpochs=15


f1_scores=[]
n_splits=5 #Number of folds (adjust as necessary)
cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# print(oneHotEncoding(()))


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

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%, Test F1 Score: {f1}")
