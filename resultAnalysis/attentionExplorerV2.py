import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
import os
import hapi

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json
import plotly.graph_objects as go
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
        #configFilePath="/home/tristanb/scratch/configFiles/"
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

        #Linear projections for Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        #Split into heads
        Q = Q.view(batch_size, seq_length, self.num_heads, input_dim // self.num_heads)
        K = K.view(batch_size, seq_length, self.num_heads, input_dim // self.num_heads)
        V = V.view(batch_size, seq_length, self.num_heads, input_dim // self.num_heads)

        #Transpose to (batch, heads, seq_len, feature_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        #Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (input_dim ** 0.5)
        attention_weights = self.softmax(scores)

        #Weighted sum of values
        weighted_sum = torch.matmul(attention_weights, V)

        #Concatenate heads and apply linear projection
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).contiguous()
        weighted_sum = weighted_sum.view(batch_size, seq_length, input_dim)

        #Output linear layer
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




def loadExample(csvPath=None,configPath=None):
    if configPath!=None:
        #Means that we're doing a custom call
        data=pd.read_csv(csvPath)
        wv,tr=data.iloc[:,0],data.iloc[:,1]
        input_data=torch.tensor(np.stack([wv, tr], axis=1), dtype=torch.float32)

        #add a batch dimension (1, since it's one example)
        input_data=input_data.unsqueeze(0)

        detect=detectionModel()
        detect.load_state_dict(torch.load(
            r"C:\Users\Tristan\Downloads\HyPCAR3\flexibleDetectionModel.pt",
            weights_only=True))
        detect.eval()

        model=abundanceModel()
        model.load_state_dict(torch.load(
            r"C:\Users\Tristan\Downloads\HyPCAR3\setWeightAbundance.pt",
            weights_only=True))
        model.eval()

        with torch.no_grad():
            detectionOutput=detect(input_data)
            predAbun,uncertainty,attentionWeights=model(input_data,detectionOutput)

        attn=attentionWeights.cpu().numpy()[0]         #(H, Q, K)
        molecules = ["O2","N2","H2","CO2","H2O","CH4","NH3"]
        print("PRED")
        for idx, mol in enumerate(molecules):
            print(f"{mol}: {predAbun[0][idx]*100:.2f}%")
            print(f"Uncertainty: {uncertainty[0][idx]}")
        H,Q,K=attn.shape
        wv,tr=torch.tensor(wv),torch.tensor(tr)
        N=wv.shape[0]

        bins=np.floor(np.arange(N) * K / N).astype(int).clip(0, K-1)
        return attn, wv, tr, bins, molecules,configPath



    if csvPath==None:
        possibleFolders=list(os.listdir(r"C:\Users\Tristan\Downloads\HyPCAR3\data"))
        possibleFolders.remove("None")
        folder=random.choice(possibleFolders)
        folderPath=os.path.join(r"C:\Users\Tristan\Downloads\HyPCAR3\data",folder)

        file=random.choice(list(os.listdir(folderPath)))
        newCsvPath=os.path.join(folderPath,file)
        data=pd.read_csv(newCsvPath)
    else:
        data=pd.read_csv(csvPath)

    wv,tr=list(map(wavelengthFilter,data.iloc[:,0])),list(data.iloc[:,1])
    input_data=torch.tensor(np.stack([wv, tr], axis=1), dtype=torch.float32)

    #add a batch dimension (1, since it's one example)
    input_data=input_data.unsqueeze(0)

    detect=detectionModel()
    detect.load_state_dict(torch.load(
        r"C:\Users\Tristan\Downloads\HyPCAR3\flexibleDetectionModel.pt",
        weights_only=True))
    detect.eval()

    model=abundanceModel()
    model.load_state_dict(torch.load(
        r"C:\Users\Tristan\Downloads\HyPCAR3\setWeightAbundance.pt",
        weights_only=True))
    model.eval()
    import time
    start=time.time()
    with torch.no_grad():
        detectionOutput=detect(input_data)
        predAbun,uncertainty,attentionWeights=model(input_data,detectionOutput)
    print(time.time()-start)
    attn=attentionWeights.cpu().numpy()[0]         #(H, Q, K)

    H,Q,K=attn.shape
    wv,tr=torch.tensor(wv),torch.tensor(tr)
    N=wv.shape[0]

    bins=np.floor(np.arange(N) * K / N).astype(int).clip(0, K-1)

    molecules = ["O2","N2","H2","CO2","H2O","CH4","NH3"]

    if csvPath==None:
        file=file.removesuffix(".csv")
        file+=".txt"
        print(file)
        config=os.path.join(r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles",file)
    else:
        
        file=os.path.basename(csvPath)
        file=file.removesuffix(".csv")
        file+=".txt"
        config=os.path.join(r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles",os.path.basename(file))

    labels=getAbundances(config)
    
    print("REAL")
    for idx, mol in enumerate(molecules):
        print(f"{mol}: {labels[idx]*100:.2f}%")
    print("\nPRED")
    for idx, mol in enumerate(molecules):
        print(f"{mol}: {predAbun[0][idx]*100:.2f}%")
        print(f"{mol}: {uncertainty[0][idx]} Uncertainty")


    with open(r"C:\Users\Tristan\Downloads\HyPCAR3\visuals\tempData.csv","w") as f:
        for i in range(len(wv)):
            f.write(str(wv[i].item())+","+str(tr[i].item())+"\n")
        print("DONE")
        

    return attn, wv, tr, bins, molecules,config

def getAtmoVal(config):
    lines=[]
    with open(config) as f:
        for line in f:
            lines.append(line)

    atmoInfo=lines[54]
    atmoInfo=atmoInfo.removeprefix("<ATMOSPHERE-LAYER-1>")
    atmoInfo=atmoInfo.split(",")
    return list(map(float,atmoInfo[:2]))

def convertCMtoUM(cm):#Convrets cm^-1 to microns
    return 10000/cm 
def read_hapi_table(mol, db_path='HAPI_DB'):
    """
    Given a HAPI table name (e.g. "O2_HITRAN"), this:
      1) Ensures HAPI has fetched the .dat/.header
      2) Reads the .header JSON to compute column widths
      3) Uses pandas.read_fwf to parse the .dat exactly
    
    Returns a DataFrame with all fields from the .dat.
    """

    
    headerPath="C:\\Users\\Tristan\\Downloads\\HyPCAR3\\HAPI_DB\\"+ f"{mol}.header"
    dataPath="C:\\Users\\Tristan\\Downloads\\HyPCAR3\\HAPI_DB\\"+ f"{mol}.data"
    #2) load header JSON
    with open(headerPath, 'r') as f:
        header = json.load(f)

    order= header['order']
    pos= header['position']    #start indices (0-based)

    line_len = None
    with open(dataPath, 'r') as f_dat:
        for line in f_dat:
            if line.strip() and not line.startswith('#'):
                line_len = len(line.rstrip('\n'))
                break
    if line_len is None:
        raise ValueError(f"No data lines found in {dataPath}")


    widths = []
    for i, field in enumerate(order):
        start = pos[field]
        if i < len(order) - 1:
            next_start = pos[order[i+1]]
            width = next_start - start
        else:
            width = line_len - start  #rest of line
        widths.append(width)


    df = pd.read_fwf(
        dataPath,
        widths=widths,
        names=order,
        comment='#',
        header=None
    )
    return df

def loadHITRANData(molecule):

    #Calculate Wavenumber bounds from spectrum (cm⁻¹)
    nu_min = 1e4 / wav.max().item()
    nu_max = 1e4 / wav.min().item()
    molIds={'H2O': 1, 'CO2': 2, 'O3': 3, 'N2O': 4, 'CO': 5, 'CH4': 6, 'O2': 7, 'NO': 8, 'SO2': 9, 'NO2': 10, 'NH3': 11, 'HNO3': 12, 'OH': 13, 'HF': 14, 'HCl': 15, 'HBr': 16, 'HI': 17, 'ClO': 18, 'OCS': 19, 'H2CO': 20, 'HOCl': 21, 'N2': 22, 'HCN': 23, 'CH3Cl': 24, 'H2O2': 25, 'C2H2': 26, 'C2H6': 27, 'PH3': 28, 'COF2': 29, 'SF6': 30, 'H2S': 31, 'HCOOH': 32, 'HO2': 33, 'O': 34, 'ClONO2': 35, 'NO+': 36, 'HOBr': 37, 'C2H4': 38, 'CH3OH': 39, 'CH3Br': 40, 'CH3CN': 41, 'CF4': 42, 'C4H3': 43, 'HC3N': 44, 'H2': 45, 'CS': 46, 'SO3': 47, 'C2N2': 48, 'COC12': 49, 'SO': 50, 'CH3F': 51, 'GeH4': 52, 'CS2': 53, 'CH3I': 54, 'NF3': 55}
    mol_id=molIds[molecule]
    hapi.fetch(molecule, mol_id, 1, nu_min, nu_max)

def getLineCenters(mol,nuMin,nuMax):
    molIds={'H2O':1,'CO2':2,'O3':3,'N2O':4,'CO':5,
              'CH4':6,'O2':7,'NO':8,'SO2':9,'NO2':10,
              'NH3':11,'HNO3':12,'OH':13,'HF':14}

    loadHITRANData(mol)
    pressure,temp=getAtmoVal(config)
    nuGrid=np.linspace(nuMin, nuMax, 784)  
    nu,coef=hapi.absorptionCoefficient_Lorentz(SourceTables=mol,HITRAN_units=False,WavenumberGrid=nuGrid)
    nu,transMol=hapi.transmittanceSpectrum(nu, coef)

    nu=1e4/nu#Convert into um
    os.remove(r"C:\Users\Tristan\Downloads\HyPCAR3"+f"\\{mol}.data")
    os.remove(r"C:\Users\Tristan\Downloads\HyPCAR3"+f"\\{mol}.header")
    return nu,transMol

        
'''

REAL
O2: 4.49%
N2: 57.85%
H2: 0.00%
CO2: 21.53%
H2O: 16.13%
CH4: 0.00%
NH3: 0.00%

Fine Tuned:
O2: 11.54%
O2: 0.07302048057317734 Uncertainty
N2: 54.74%
N2: 0.23366165161132812 Uncertainty
H2: 0.00%
H2: 0.1850724071264267 Uncertainty
CO2: 18.56%
CO2: 0.05876367911696434 Uncertainty
H2O: 15.17%
H2O: 0.04903728514909744 Uncertainty
CH4: 0.00%
CH4: 0.0732705295085907 Uncertainty
NH3: 0.00%
NH3: 0.12598568201065063 Uncertainty

PRED
O2: 12.21%
N2: 48.44%
H2: 0.00%
CO2: 21.92%
H2O: 17.43%
CH4: 0.00%
NH3: 0.00%

'''



#attn,wav,tr,bins,molecules,config=loadExample(r"C:\Users\Tristan\Downloads\HyPCAR3\data\B\B_8104.csv")
#attn,wav,tr,bins,molecules,config=loadExample()
#attn,wav,tr,bins,molecules,config=loadExample(r"C:\Users\Tristan\Downloads\HyPCAR3\earthTransmittance.csv",r"C:\Users\Tristan\Downloads\HyPCAR3\earthConfigTemplate.txt")
attn,wav,tr,bins,molecules,config=loadExample(r"C:\Users\Tristan\Downloads\HyPCAR3\marsSpectrum.csv",r"C:\Users\Tristan\Downloads\HyPCAR3\earthConfigTemplate.txt")

H,Q,K=attn.shape

#Calculate the wavenumber bounds from the spectrum
nu_min=1e4/wav.max().item()
nu_max=1e4/wav.min().item()

#print(getAtmoVal(config))


#Only do this if you don't have data
sim = {}
#for mol in molecules:
#    wav_sim, tr_sim = getLineCenters(mol, nu_min, nu_max)
#    sim[mol] = (wav_sim, tr_sim)
#    #Save the data, so we don't have to keep doing this
#    folder=r"C:\Users\Tristan\Downloads\HyPCAR3\resultAnalysis\molecularTransmittance"
#    fn=os.path.join(folder,f"{mol}.csv")
#    with open(fn,"w") as f:
#        for i in range(len(wav_sim)):
#            f.write(str(wav_sim[i])+","+str(tr_sim[i])+"\n")
    
#B_8104.txt is really good example
#C:\Users\Tristan\Downloads\HyPCAR3\data\B\B_8104.csv
sim={}
fingerprints={}
for mol in molecules:
    folder=r"C:\Users\Tristan\Downloads\HyPCAR3\resultAnalysis\molecularTransmittance"
    fn=os.path.join(folder,f"{mol}.csv")
    molData=pd.read_csv(fn,header=None)

    wv2,tr2=molData.iloc[:,0],molData.iloc[:,1]
    sim[mol]=(wv2,tr2)

    fp=1-np.array([tr2[bins==k].mean() for k in range(K)])
    fingerprints[mol]=fp


#Corrleate average attention (over queries) with molecular absorption features
head_attn=attn.mean(axis=1) if attn.ndim==3 else attn   #shape (H, K)

corr_dict = {}
for mol,fp in fingerprints.items():
    #If the spectrum is nearly flat (i.e. for molecules like H2, O2, N2),
    #the fingerprint's variance will be near zero
    if np.std(fp)<1e-6:
        #Mark correlation as NaN or a default value (here NaN) because there's no variation
        corr_dict[mol]=[np.nan]*H
    else:
        mol_corr=[]
        for h in range(H):
            if np.std(head_attn[h, :])<1e-6:
                mol_corr.append(np.nan)
            else:
                mol_corr.append(np.corrcoef(head_attn[h,:],fp)[0, 1])
        corr_dict[mol]=mol_corr

corr=pd.DataFrame(corr_dict, index=[f"Head {h+1}" for h in range(H)])

#--- Build Dash App ---
app = Dash(__name__)
app.layout = html.Div([
    html.H2("Molecule-First Attention Explorer"),
    html.Div([
        html.Label("Molecule:"),
        dcc.Dropdown(
            id='mol-dd',
            options=[{'label': m, 'value': m} for m in molecules],
            value=molecules[0],
            clearable=False
        ),
        html.Br(),
        html.Label("Head (sorted by correlation):"),
        dcc.Dropdown(id='head-dd', clearable=False),
        html.Br(),
        html.Label("Query:"),
        dcc.Dropdown(
            id='query-dd',
            options=[{'label': f'Query {q+1}', 'value': q} for q in range(Q)],
            value=0,
            clearable=False
        ),
    ], style={'width':'25%', 'display':'inline-block', 'verticalAlign':'top', 'padding':'20px'}),
    html.Div([
        dcc.Graph(id='heatmap', style={'height':'250px'}),
        dcc.Graph(id='bar-chart', style={'height':'250px'}),
        dcc.Graph(id='binned-scatter', style={'height':'400px'}),
    ], style={'width':'70%', 'display':'inline-block', 'padding':'20px'}),
])

#--- Callback to update Head dropdown based on the selected molecule ---
@app.callback(
    Output('head-dd','options'),
    Output('head-dd','value'),
    Input('mol-dd','value'),
)
def update_heads(mol):
    #Sort the heads for the selected molecule by descending correlation
    scores=pd.Series(corr[mol], index=[f"Head {h+1}" for h in range(H)]).sort_values(ascending=False)
    opts=[
        {'label': f"{head} (r={scores[head]:.2f})", 'value': int(head.split()[1]) - 1}
        for head in scores.index
    ]
    return opts,opts[0]['value']

#--- Main visualization callback ---
@app.callback(
    Output('heatmap','figure'),
    Output('bar-chart','figure'),
    Output('binned-scatter','figure'),
    Input('head-dd','value'),
    Input('query-dd','value'),
    Input('mol-dd','value'),
)
def update_plots(head, query, mol):
    vec=attn[head, query, :]

    #1) Heatmap: show attention vector as a one-row heatmap.
    hm=px.imshow(
        vec.reshape(1, K),
        x=[f'K{k}' for k in range(K)],
        y=[f'H{head+1}, Q{query+1}'],
        color_continuous_scale='viridis', zmin=0, zmax=1,
        labels={'x': 'Token', 'y': '', 'color': 'Weight'}
    )
    hm.update_yaxes(showticklabels=False)

    #2) Bar chart showing attention per token.
    bc=px.bar(
        x=[f'K{k}' for k in range(K)],
        y=vec,
        range_y=[0,1],
        labels={'x': 'Token', 'y': 'Attention'},
        title=f'Head {head+1}, Query {query+1}'
    )

    #3) Binned scatter plot:
    #Here we plot the global measured spectrum (wav, tr) colored by the attention weight (using bins)
    colors=vec[bins]
    sc=px.scatter(
        x=wav, y=tr,
        color=colors,
        color_continuous_scale='viridis',
        labels={'color':'Token Weight'},
        title=f'Spectrum & Attention vs {mol}'
    ).update_traces(marker={'size':6})
    
    #4) Overlay the full molecule envelope (from simulated HITRAN data),
    #including the head’s correlation with the absorption fingerprint.
    wav_env, tr_env=sim[mol]
    sc.add_trace(go.Scatter(
        x=wav_env, y=tr_env,
        mode='lines',
        name=f'{mol} envelope (r={corr.loc[f"Head {head+1}", mol]:.2f})',
        line=dict(dash='dash', width=2),
        opacity=0.6
    ))
    
    return hm, bc, sc

if __name__=='__main__':
    app.run(debug=True)