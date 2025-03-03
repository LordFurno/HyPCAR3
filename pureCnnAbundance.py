import torch
from torch.utils.data import DataLoader,Dataset,random_split
import pandas as pd
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self,samples):#samples contain a listt of all file paths
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
        return abundances,attention_weights





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

            outputs,attentionWeights=model(data,detectionOutput)

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
            with open("pureCNN.txt",'a') as f:
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

                    outputs,attentionWeights=model(data,moleculeDetection)

                    loss=criterion(outputs,labels)


                    val_loss+=loss.item()
                    validation_KL_loss+=kl_divergence(outputs,labels)
                    validation_top_k+=topKAccuracy(outputs,labels,1)
                    validation_cross_entropy+=customCrossEntropy(outputs,labels)

                

            valLoss=val_loss/len(validationDataloader)
            valKL=validation_KL_loss/len(validationDataloader)
            valTopK=validation_top_k/len(validationDataloader)
            valCrossEntropy=validation_cross_entropy/len(validationDataloader)
            with open("pureCNN.txt",'a') as f:
                f.write(f"Validation Loss: {valLoss}, KL Divergence: {valKL}, Top-K Accuracy: {valTopK}, Cross Entropy: {valCrossEntropy}"+"\n")
            print(f"Validation Loss: {valLoss}, KL Divergence: {valKL}, Top-K Accuracy: {valTopK}, Cross Entropy: {valCrossEntropy}")
        break#Just 1 validation fold

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


            loss=criterion(outputs,labels)


            test_loss+=loss.item()
            test_kl_loss+=kl_divergence(outputs,labels)
            test_top_k+=topKAccuracy(outputs,labels,1)
            test_cross_entropy+=customCrossEntropy(outputs,labels)
            

        test_loss=test_loss/len(testingDataloader)
        testKL=test_kl_loss/len(testingDataloader)
        testTopK=test_top_k/len(testingDataloader)
        testCrossEntropy=test_cross_entropy/len(testingDataloader)

    with open("pureCNN.txt",'a') as f:
        f.write(f"Testing Loss: {test_loss}, KL Divergence: {testKL}, Top-K Accuracy: {testTopK}, Cross Entropy: {testCrossEntropy}"+"\n")
    # print(f"Testing Loss: {test_loss}, KL Divergence: {testKL}, Top-K Accuracy: {testTopK}, Cross Entropy: {testCrossEntropy}")
