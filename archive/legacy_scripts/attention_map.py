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
from math import ceil
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
        # configFilePath="configFiles/"
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

        self.fcDetect=nn.Linear(7, 64)


        self.conv1=nn.Conv1d(in_channels=2, out_channels=256, kernel_size=5, stride=2)
        self.bn1=nn.BatchNorm1d(256)
        self.pool1=nn.MaxPool1d(2)

        self.conv2=nn.Conv1d(in_channels=256,out_channels=512,kernel_size=7,stride=1)
        self.bn2=nn.BatchNorm1d(512)
        self.pool2=nn.MaxPool1d(2)

        self.conv3=nn.Conv1d(in_channels=512,out_channels=256,kernel_size=5,stride=2)
        self.bn3=nn.BatchNorm1d(256)
        self.pool3=nn.MaxPool1d(2)

        self.conv4=nn.Conv1d(in_channels=256,out_channels=64,kernel_size=2,stride=2)
        self.bn4=nn.BatchNorm1d(64)
        self.pool4=nn.MaxPool1d(2)

        self.dropout1=nn.Dropout(0.329397809173006)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.attention=MultiHeadAttention(input_dim=64, num_heads=16)


        self.fc_combined=nn.Linear(128, 128)#Combines both input branches (detection + data)


        self.flatten=nn.Flatten()


        self.dropout2=nn.Dropout(0.502219550897328)
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
    model.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\finalBaseAbundance.pt",weights_only=True))
    model=model.to(device)


    detect=detect=detectionModel()
    detect.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\flexibleDetectionModel.pt",weights_only=True))
    detect=detect.to(device)


    model.eval()
    detect.eval()
    with torch.no_grad():
        for batch in tqdm(testingDataloader):
            data,labels,configs=batch
            data=data.to(device)
            labels=labels.to(device)


            detectionOutput=detect(data)
            predAbun,uncertainties,attentionWeights=model(data, detectionOutput)
            #attentionWeights is now (1, 16, 5, 5)

            #Code for plotly thing
            #attn=attentionWeights.cpu().numpy()[0] 


            #wav=data[0,:,0].cpu().numpy()
            #tr =data[0,:,1].cpu().numpy()

            #N=wav.shape[0]
            #bins=np.floor(np.arange(N) * 5 / N).astype(int)
            #bins=np.clip(bins, 0, 4)
            #break


            raw=attentionWeights[0]                        

            
            N=data.shape[1]
            bins=np.floor(np.arange(N) * 5 / N).astype(int)
            bins=np.clip(bins, 0, 4)

            wav=data[0,:,0].cpu().numpy()
            tr=data[0,:,1].cpu().numpy()


            fig,axes = plt.subplots(4,4, figsize=(16,12))
            axes=axes.flatten()
            for h in range(raw.shape[0]):
                #per-head token weights
                w=raw[h].mean(dim=0).cpu().numpy()  # (5,)
                colors=w[bins]                      # (N,)
                sc=axes[h].scatter(wav, tr, c=colors, cmap="viridis", s=20)
                axes[h].set_title(f"Head {h+1}")
                axes[h].set_xticks([])
                axes[h].set_yticks([])
                plt.colorbar(sc, ax=axes[h], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()




            print("REAL")
            for idx, mol in enumerate(molecules):
                print(f"{mol}: {labels[0][idx]*100:.2f}%")
            print("\nPRED")
            for idx, mol in enumerate(molecules):
                print(f"{mol}: {predAbun[0][idx]*100:.2f}%")
            print(f"\nUncertainties: {uncertainties[0].tolist()}")

            break
    
# import dash
# from dash import dcc, html, Input, Output
# import plotly.express as px
# H, Q, K = attn.shape  # expected (16,5,5)
# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H2("HyPCAR Attention Explorer"),
#     html.Div([
#         html.Label("Head:"),
#         dcc.Dropdown(
#             id='head-dd',
#             options=[{'label': f'Head {h+1}', 'value': h} for h in range(H)],
#             value=0
#         ),
#         html.Label("Query:"),
#         dcc.Dropdown(
#             id='query-dd',
#             options=[{'label': f'Query {q}', 'value': q} for q in range(Q)],
#             value=0
#         ),
#     ], style={'width':'20%', 'display':'inline-block', 'verticalAlign':'top'}),

#     html.Div([
#         dcc.Graph(id='heatmap'),
#         dcc.Graph(id='bar-chart'),
#         dcc.Graph(id='binned-scatter'),
#     ], style={'width':'75%', 'display':'inline-block', 'padding':'0 20'}),
# ])

# @app.callback(
#     Output('heatmap', 'figure'),
#     Output('bar-chart', 'figure'),
#     Output('binned-scatter', 'figure'),
#     Input('head-dd','value'),
#     Input('query-dd','value'),
# )
# def update_plots(head, query):
#     # slice out the exact 5-token weights
#     vec = attn[head, query, :]  # shape (5,)

#     # 1) heatmap (1×5)
#     hm = px.imshow(
#         vec.reshape(1,K),
#         labels={'x':'Key Token','y':'','color':'Weight'},
#         x=[f'K{k}' for k in range(K)],
#         y=[f'H{head+1},Q{query}'],
#         color_continuous_scale='viridis'
#     )
#     hm.update_yaxes(showticklabels=False)

#     # 2) bar chart
#     bc = px.bar(
#         x=[f'K{k}' for k in range(K)],
#         y=vec,
#         labels={'x':'Key Token','y':'Attention Weight'},
#         title=f'Head {head+1}, Query {query}'
#     )

#     # 3) binned scatter
#     colors = vec[bins]  # map each of the N points → its token weight
#     sc = px.scatter(
#         x=wav, y=tr, color=colors,
#         color_continuous_scale='viridis',
#         labels={'color':'Token Weight'},
#         title='Spectrum colored by exact token-weights'
#     )
#     sc.update_traces(marker={'size':6})

#     return hm, bc, sc

# if __name__ == "__main__":
#     app.run(debug=True)