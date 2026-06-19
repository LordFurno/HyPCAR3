import torch
from torch.utils.data import DataLoader,Dataset,random_split

import torch.nn as nn
import torch.nn.functional as F

class flexibleDetectionModel(nn.Module):
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

class flexibleAbundanceModel(nn.Module):#CHange to output uncertainty as well
    def __init__(self):
        super().__init__()

        self.fcDetect=nn.Linear(7, 64)


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

        self.attention=MultiHeadAttention(input_dim=32, num_heads=8)


        self.fc_combined=nn.Linear(96, 128)#Combines both input branches (detection + data)


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



def runFlexibleDetection(data):
    '''
    This function runs the detection model to find what molecules are present.

    Inputs
    ------
    data: The data as a stacked tensor of wavelength and transmittance.
    
    Returns
    -------
    probability: A tensor containing the probability for each molecules presence. 
    '''
    moleculeDetection=flexibleDetectionModel()
    moleculeDetection.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\flexibleDetectionModel.pt",weights_only=True))
    moleculeDetection.eval()
    with torch.no_grad():
        detectionOutput=moleculeDetection(data)
    return detectionOutput

def runDetection(data):
    '''
    This function runs the detection model to find what molecules are present.

    Inputs
    ------
    data: The data as a stacked tensor of wavelength and transmittance.
    
    Returns
    -------
    probability: A tensor containing the probability for each molecules presence. 
    '''
    moleculeDetection=detectionModel()
    moleculeDetection.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\detectionModel.pt",weights_only=True))
    moleculeDetection.eval()
    with torch.no_grad():
        detectionOutput=moleculeDetection(data)
    return detectionOutput


def runFlexibleAbundance(data,returnAttention=False):
    '''
    This function runs the flexibleAbundance model. It will return its output as a list.

    Inputs
    ------
    data: The data as a stacked tensor of wavelength and transmittance.
    returnAttention: Whether or not to return attentionweights. Default is false.
    
    Returns
    -------
    abundances: A tensor containing HyPCAR's predicted molecule abundances.
    '''
    curModel=flexibleAbundanceModel()
    moleculeDetection=flexibleDetectionModel()


    curModel.load_state_dict(torch.load("flexibleBaseAbundance.pt",weights_only=True))
    moleculeDetection.load_state_dict(torch.load("flexibleDetectionModel.pt",weights_only=True))

    curModel.eval()
    moleculeDetection.eval()


    with torch.no_grad():
        detectionOutput=moleculeDetection(data)
        abundanceOutput,uncertainty,attentionWeights=curModel(data,detectionOutput)


    if returnAttention:
        return abundanceOutput,attentionWeights
    else:
        return abundanceOutput