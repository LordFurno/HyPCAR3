import torch
from torch.utils.data import DataLoader,Dataset,random_split
import pandas as pd
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time


#55922394
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

    detect.load_state_dict(torch.load("/home/tristanb/projects/def-pjmann/tristanb/flexibleDetectionModel.pt",weights_only=True))
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
    def __init__(self,samples):#samples contain a listt of all file paths
        self.samples=samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index][0],self.samples[index][1]

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

class abundanceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fcDetect=nn.Linear(7, 64)

        self.conv1=nn.Conv1d(in_channels=2, out_channels=128, kernel_size=5, stride=2)
        self.bn1=nn.BatchNorm1d(128)
        self.pool1=nn.MaxPool1d(2)

        self.conv2=nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2)
        self.bn2=nn.BatchNorm1d(128)
        self.pool2=nn.MaxPool1d(2)

        self.conv3=nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.bn3=nn.BatchNorm1d(64)
        self.pool3=nn.MaxPool1d(2)

        self.conv4=nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.bn4=nn.BatchNorm1d(32)
        self.pool4=nn.MaxPool1d(2)

        self.dropout1=nn.Dropout(0.4)
        self.global_pool=nn.AdaptiveAvgPool1d(1)

        self.fc_combined=nn.Linear(96, 128)  # Combines both input branches (detection + data)

        self.dropout2=nn.Dropout(0.75)
        self.fc2=nn.Linear(128, 64)
        self.fc3=nn.Linear(64, 32)

        self.fc4=nn.Linear(32, 7)  # Abundance output
        self.fc_uncertainty=nn.Linear(32, 7)  # Uncertainty output

    def forward(self, x, detectionOutput):
        detectionOutput=F.relu(self.fcDetect(detectionOutput))

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
        x=self.global_pool(x).squeeze(-1)

        combined=torch.cat((x, detectionOutput), dim=1)
        combined=F.relu(self.fc_combined(combined))
        combined=F.relu(self.fc2(combined))
        combined=F.relu(self.fc3(combined))

        logits=self.fc4(combined)
        abundances=F.softmax(logits, dim=1)

        uncertaintyRaw=self.fc_uncertainty(combined)
        uncertainties=F.softplus(uncertaintyRaw)

        return abundances,uncertainties



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
        configFolderPath=os.path.join(os.environ["SLURM_TMPDIR"],"configFiles")

        files=[]
        for path in os.listdir(curFolderPath):
            #Need to get molecule abundances as well, this means that for each file, I need to go to the config file
            #Then extract the abundances there
            #This is how I will get the one-hot vector for the presence
            #Will just write a functionn

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



    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detect=detectionModel()
    detect=detect.to(device)

    print(device)


    numEpochs=15



    '''
    Consider pre-loading the data set before training even beings. This would make it so that I only ever have to load
    the data once. Then the batches will be super fast, even for a seperate validation split.
    '''

    start1=time.time()
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



    criterion=nn.MSELoss()

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

            loss=criterion(outputs,labels)
            running_loss+=loss.item()
            running_KL_loss+=kl_divergence(outputs,labels)
            running_top_k+=topKAccuracy(outputs,labels,1)
            running_cross_entropy+=customCrossEntropy(outputs,labels)


            loss.backward()

            optimizer.step()

        trainingLoss=running_loss/len(trainingDataloader)
        klLoss=running_KL_loss/len(trainingDataloader)
        topKAcc=running_top_k/len(trainingDataloader)
        crossEntropy=running_cross_entropy/len(trainingDataloader)


        #Instead of weighted accuracy, use top-k accuracy and R^2 value.
        with open("naiveCNN.txt",'a') as f:
            f.write(f"Epoch {epoch+1}, Loss: {trainingLoss}, KL Divergence: {klLoss}, Top-K Accuracy: {topKAcc}, Cross Entropy: {crossEntropy}"+"\n")

        print(f"Epoch {epoch+1}, Loss: {trainingLoss}, KL Divergence: {klLoss}, Top-K Accuracy: {topKAcc}, Cross Entropy: {crossEntropy}")

        model.eval()
        with torch.no_grad():
            val_loss=0
            validation_KL_loss=0.0

            validation_top_k=0.0
            validation_cross_entropy=0.0

            counter=0
            for index,batch in enumerate(validationDataloader):
                data,labels,config=batch

                data=data.to(device)
                labels=labels.to(device)

                optimizer.zero_grad()

                moleculeDetection=detectMolecules(data)

                outputs,uncertainties,attentionWeights=model(data,moleculeDetection)

                loss=criterion(outputs,labels)


                val_loss+=loss.item()
                validation_KL_loss+=kl_divergence(outputs,labels)
                validation_top_k+=topKAccuracy(outputs,labels,1)
                validation_cross_entropy+=customCrossEntropy(outputs,labels)



            valLoss=val_loss/len(validationDataloader)
            valKL=validation_KL_loss/len(validationDataloader)
            valTopK=validation_top_k/len(validationDataloader)
            valCrossEntropy=validation_cross_entropy/len(validationDataloader)
        with open("naiveCNN.txt",'a') as f:
            f.write(f"Validation Loss: {valLoss}, KL Divergence: {valKL}, Top-K Accuracy: {valTopK}, Cross Entropy: {valCrossEntropy}"+"\n")
        print(f"Validation Loss: {valLoss}, KL Divergence: {valKL}, Top-K Accuracy: {valTopK}, Cross Entropy: {valCrossEntropy}")

    torch.save(model.state_dict(), "naiveCNN.pt")
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
            data,labels=batch

            data=data.to(device)
            labels=labels.to(device)


            moleculeDetection=detectMolecules(data)

            outputs,uncertainties,attentionWeights=model(data,moleculeDetection)


            loss=criterion(outputs,labels)


            test_loss+=loss.item()
            test_kl_loss+=kl_divergence(outputs,labels)
            test_top_k+=topKAccuracy(outputs,labels,1)
            test_cross_entropy+=customCrossEntropy(outputs,labels)


        test_loss=test_loss/len(testingDataloader)
        testKL=test_kl_loss/len(testingDataloader)
        testTopK=test_top_k/len(testingDataloader)
        testCrossEntropy=test_cross_entropy/len(testingDataloader)

    with open("naiveCNN.txt",'a') as f:
        f.write(f"Testing Loss: {test_loss}, KL Divergence: {testKL}, Top-K Accuracy: {testTopK}, Cross Entropy: {testCrossEntropy}"+"\n")
    # print(f"Testing Loss: {test_loss}, KL Divergence: {testKL}, Top-K Accuracy: {testTopK}, Cross Entropy: {testCrossEntropy}")