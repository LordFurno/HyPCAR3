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
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import seaborn as sns
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
def visualize_attention(
    attention_weights,    # PyTorch tensor of shape (num_heads, H, W)
    wavelength,           # 1D numpy array or tensor of wavelengths
    transmittance,        # 1D numpy array or tensor of transmittance values
    molecules=None,       # List of molecule names (optional)
    labels=None,          # Ground truth abundances (optional)
    predAbun=None,        # Predicted abundances (optional)
    uncertainties=None    # Uncertainty values (optional)
):
    # Ensure wavelength and transmittance are numpy arrays
    if torch.is_tensor(wavelength):
        wavelength = wavelength.cpu().numpy()
    if torch.is_tensor(transmittance):
        transmittance = transmittance.cpu().numpy()
    
    # Determine the sequence length from the wavelength array
    seq_length = len(wavelength)
    
    # If your attention weights are of shape (num_heads, H, W),
    # you might need to add a batch dimension for the interpolation function.
    # Here we assume batch size = 1.
    # attn = attention_weights.unsqueeze(0)  # shape: (1, num_heads, H, W)
    
    # Interpolate attention weights to match the sequence length (both dimensions)
    upsampled_weights = torch.nn.functional.interpolate(
        attention_weights, size=(seq_length, seq_length), mode="bilinear", align_corners=False
    )
    
    # Aggregate across heads (average)
    aggregated_attention = upsampled_weights[0].mean(dim=0).cpu().numpy()  # shape: (seq_length, seq_length)
    # Sum attention values for each wavelength (across rows)
    attention_sums = aggregated_attention.sum(axis=1)
    
    # Print details if provided
    if molecules is not None and labels is not None:
        print("REAL")
        for index, molecule in enumerate(molecules):
            print(f"{molecule}: {labels[0][index]*100:.2f}%")
        print()
    
    if molecules is not None and predAbun is not None:
        print("PRED")
        for index, molecule in enumerate(molecules):
            print(f"{molecule}: {predAbun[0][index]*100:.2f}%")
    
    if uncertainties is not None:
        print(f"Uncertainties: {uncertainties[0].tolist()}")
    
    # Optionally print raw attention weights if desired
    print(f"Attention Weights: {attention_weights[0].tolist()}")
    
    # Plot scatterplot of wavelength vs. transmittance colored by aggregated attention intensity
    plt.figure(2)
    plt.figure(figsize=(10, 8))
    plt.scatter(wavelength, transmittance, c=attention_sums, cmap="viridis", s=100)
    plt.colorbar(label="Attention Intensity (Aggregated Across Heads)")
    plt.title("Attention on Wavelength vs. Transmittance (Aggregated Heads)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.savefig(r"C:\Users\Tristan\Downloads\HyPCAR3\visuals\earthAttentionMap.png")
    plt.show()
    
    # Create heatmaps for each attention head
    attention_weights=attention_weights[0]
    num_heads = attention_weights.shape[0]
    print(attention_weights.shape)
    plt.figure(3)
    plt.figure(figsize=(10, 5))
    
    # Adjust grid based on the number of heads (here, assuming up to 8 heads for a 2x4 grid)
    # print(num_heads)
    for i in range(num_heads):
        ax = plt.subplot(2, 4, i + 1)
        # Select the attention weights for the current head, resize if needed
        head_weights = attention_weights[i].cpu().numpy()
        sns.heatmap(head_weights, annot=True, cmap='viridis', ax=ax, cbar=False)
        ax.set_title(f'Head {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(r"C:\Users\Tristan\Downloads\HyPCAR3\visuals\earthAttentionHeads.png")
    plt.show()

aModel=abundanceModel()
model=detectionModel()

#Load the saved model weights
model.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\detectionModel.pt",weights_only=True))
aModel.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\pureCNN.pt",weights_only=True))

# filePath=r"C:\Users\Tristan\Downloads\HyPCAR\table_K2-18-b-Madhusudhan-et-al.-2023 (2).csv"#File path for the data
# filePath=r"C:\Users\Tristan\Downloads\HyPCAR\table_HAT-P-18-b-Fu-et-al.-2022 (1).csv"#File path for the data
# data=pd.read_csv(filePath)

# wavelength=data["CENTRALWAVELNG"]
# transmittance=data["PL_TRANDEP"]

# transmittance=[1-(t/100) for t in transmittance]#Converts depth to transmittance.

# plt.figure(0)#Plot orginial data
# plt.plot(wavelength,transmittance)


# #Adjust the data to have length 785
# # interp_func=interp1d(np.linspace(0, 1, len(wavelength)), wavelength)
# # interp_trans=interp1d(np.linspace(0, 1, len(transmittance)), transmittance)

# #Generate 785 points evenly spaced in the range [0, 1]
# x_new=np.linspace(0, 1, 785)

# # Apply the interpolation function
# wavelength=interp_func(x_new)
# transmittance=interp_trans(x_new)



# #Apply filter
# transmittance_downsampled = savgol_filter(transmittance_downsampled, window_length=50, polyorder=5)
data=pd.read_csv(r"C:\Users\Tristan\Downloads\HyPCAR2\earthTransmittance.csv")

wavelength,transmittance=data.iloc[:,0],data.iloc[:,1]


plt.figure(1)
plt.title("Earth Transmittance")
plt.xlabel("Wavelength (um)")
plt.ylabel("Transmittance")

plt.plot(wavelength,transmittance)
plt.savefig(r"C:\Users\Tristan\Downloads\HyPCAR3\visuals\earthTransmittance.png")
input_data=torch.tensor(np.stack([wavelength, transmittance], axis=1), dtype=torch.float32)

#add a batch dimension (1, since it's one example)
input_data=input_data.unsqueeze(0)

with torch.no_grad():
    model.eval()
    aModel.eval()
    output=model(input_data)

    aOutput=aModel(input_data,output)


predicted=(output > 0.5).float()
#"O2","N2","H2","CO2","H2O","CH4","NH3"
print(f"Model Output: {output}")
print(f"Predicted Class: {predicted}")


visualize_attention(aOutput[2],wavelength,transmittance)

print(aOutput[0].tolist())
print(aOutput[1].tolist())
print(aOutput[2].tolist())
print(classifyAtmosphere(aOutput[0].tolist()[0]))
#Not that bad, look into cleaning up any noise or whawtever before interpolation
plt.show(block=True)

'''
REAL
O2: 20.9%
N2: 78.1%
H2: NOPE
CO2: 379.5 ppm
H2O: 0.35%
CH4: 1.7 ppm
NH3: NOPE

PRED:
O2: YES - 
N2: YES -13.7%
H2: NOPE  
CO2: YES -6.7%
H2O: YES -44.7%
CH4: YES -33.4%
NH3: NOPE
'''
