import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
def testnewAbundance(moleculeAbundances):
    lines=[]
    with open(r"C:\Users\Tristan\Downloads\HyPCAR3\physcicsFineTune\tempConfig.txt") as f:
        for line in f:
            lines.append(line)
    abundanceDictionary={}
    molecules=["O2","N2","H2","CO2","H2O","CH4","NH3"]
    for i in range(len(moleculeAbundances)):
        abundanceDictionary[molecules[i]]=moleculeAbundances[i]

    moleculeWeights={"O2":31.999, "N2":28.02, "H2":2.016,"CO2":44.01, "H2O":18.01528,"CH4":16.04,"NH3":17.03052 }#g/mol
    averageWeight=0
    for molecule in abundanceDictionary:
        averageWeight+=moleculeWeights[molecule]*abundanceDictionary[molecule]
    


    for i in range(50):
        atmosphereInfo=lines[54+i]
        atmosphereInfo=atmosphereInfo.removeprefix("<ATMOSPHERE-LAYER-"+str(i+1)+">")
        atmosphereInfo=atmosphereInfo.removesuffix("\n")
        atmosphereInfo=atmosphereInfo.split(",")

        atmosphereInfo[2:]=moleculeAbundances
        


        lines[54+i]="<ATMOSPHERE-LAYER-"+str(i+1)+">"+",".join(map(str,list(atmosphereInfo)))+"\n"
    
    nMolecules=7

    HITRANValues={"O2":"HIT[7]","N2":"HIT[22]","H2":"HIT[45]","CO2":"HIT[2]","H2O":"HIT[1]","CH4":"HIT[6]","NH3":"HIT[11]"}

    #Additional parameters to actually run the data properly.
    lines[42]="<ATMOSPHERE-NGAS>"+str(len(moleculeAbundances))+"\n" #Number of gases are in the atmosphere
    lines[43]="<ATMOSPHERE-GAS>"+",".join(molecules)+"\n" #What gases are in the atmosphere
    lines[44]="<ATMOSPHERE-TYPE>"+",".join(HITRANValues[mol] for mol in molecules)+"\n" #HITRAN values for each gas
    lines[45]="<ATMOSPHERE-ABUN>"+"1,"*(len(moleculeAbundances)-1)+"1"+"\n" #Molecule abunadnces. They're all 1, because abundances are defined in vertical profile
    lines[46]="<ATMOSPHERE-UNIT>"+"scl,"*(len(moleculeAbundances)-1)+"scl"+"\n" #Abundance unit
    lines[49]="<ATMOSPHERE-WEIGHT>"+str(averageWeight)+"\n" #Molecule weight of atmosphere g/mol
    lines[52]="<ATMOSPHERE-LAYERS-MOLECULES>"+",".join(molecules)+"\n" #Molecule in vertical profile
    with open(r"C:\Users\Tristan\Downloads\HyPCAR3\physcicsFineTune\tempConfig.txt","w") as f:
        f.writelines(lines)
  
    
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


def calculateLikelihood(yReal,ySim,sigma):
    '''
    This function calculates the likelhood P(A_pred|Y_real). It is a gaussian likelihood function.

    Inputs
    ------
    yReal: Real, input transmittance data
    ySim: Simulated transmittance data, calculated using the predicted abundances
    sigma: Uncertainty of how well predicted abundances translate to simulated spectra

    Returns
    -------
    likelihood: Likelihood value for given real and simulated data
    '''
    
    mse = torch.mean((yReal - ySim)**2,dim=[1,2])  #Mean squared error between real and simulated data
    #return np.exp(-mse / (2 * sigma ** 2))
    nll = mse / (2 * sigma**2)
    # nll_aggregated = torch.mean(nll, dim=0)  # shape: (B,)
    # return nll_aggregated
    return nll


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


def calculateExpectedValues(predAbun):
    #First figure out what type of atmosphere the model think it is
    #Either A1, A2, B, C or unkown
    atmosphereType=classifyAtmosphere(predAbun)

    molecules={0:"O2",1:"N2",2:"H2",3:"CO2",4:"H2O",5:"CH4",6:"NH3"}
    abundanceDict={}
    for i,val in enumerate(predAbun):
        abundanceDict[molecules[i]]=val

    H=2*abundanceDict["H2"] + 2*abundanceDict["H2O"] + 3*abundanceDict["NH3"] + 4*abundanceDict["CH4"]

    C=abundanceDict["CO2"] + abundanceDict["CH4"]

    O=2*abundanceDict["O2"] + 2*abundanceDict["CO2"] + abundanceDict["H2O"]

    N=2*abundanceDict["N2"] + abundanceDict["NH3"]

    expected={}
    if atmosphereType=="A1":
        #For Type A1 atmospheres (H-rich, mainly H2O, CH4, NH3, and H2; lacking CO2, O2)
        '''
        H > 2O + 4C
        3N < H - 2O -4C

        D = H - N - 2C
        H2O = 2O / D
        NH3 = 2N / D
        CH4 = 2C / D
        H2 = (H - 2O - 4C - 3N) / D
        '''
        D = H - N - 2*C
        expected["H2O"]=2*O / D
        expected["NH3"]=2*N / D
        expected["CH4"]=2*C / D
        expected["H2"]=(H - 2*O - 4*C - 3*N) / D

        expected["O2"]=0
        expected["N2"]=0
        expected["CO2"]=0

    elif atmosphereType=="A2":
        #For Type A2 atmospheres (H-rich, but mainly H2O, CH4, NH3, and N2; lacking CO2, O2)
        '''
        D = H + 2C + 3N + 4O
        H2O = 6O / D
        NH3 = (2H - 8C - 4O) / D
        CH4 = 6C / D
        N2 = (3N + 4C + 2O - H) / D
        '''
        D = H + 2*C + 3*N + 4*O
        expected["H2O"]=6*O / D
        expected["NH3"]=(2*H - 8*C - 4*O) / D
        expected["CH4"]=6*C / D
        expected["N2"] =(3*N + 4*C + 2*O - H) / D

        expected["O2"]=0
        expected["H2"]=0
        expected["CO2"]=0

    elif atmosphereType=="B":
        #For Type B atmospheres (O-rich, mainly O2, N2, CO2, H2O; lacking NH3, H2)
        '''
        D = H + 2O + 2N
        H2O = 2H / D
        N2 = 2N / D
        CO2 = 4C / D
        O2 = (2O - H - 4C) /D
        '''
        D = H + 2*O + 2*N
        expected["H2O"]=2*H / D
        expected["N2"]=2*N / D
        expected["CO2"]=4*C / D
        expected["O2"]=(2*O - H - 4*C) / D

        expected["NH3"]=0
        expected["H2"]=0
        expected["CH4"]=0
    elif atmosphereType=="C":
        #For Type C atmospheres (Hydrogen-poor: mainly H2O, CO2, CH4, N2; lacking NH3, H2, O2)
        #Note: In Type C, the elemental budget is H + C + O + N = 1.
        '''
        Side conditions, so no negative results
        O > 0.5H + 2C -> O2-rich with no CH4
        H > 2O + 4C -> H2 ->H2-rich with no CO2
        C > 0.25H + 0.5O -> graphite condensation with no H2O

        H2O = (H + 2O - 4C) / (H + 2O + 2N)
        CH4 = (H - 2O + 4C) / (2H + 4O + 4N)
        CO2 = (2O + 4C - H) / (2H + 4O + 4N)
        N2 = 2N / (H + 2O + 2N)
        '''

        #Denom for H2O and N2
        D1 = H + 2*O + 2*N

        #Denom for CH4 and CO2
        D2 = 2*H + 4*O + 4*N  

        expected["H2O"] = (H + 2*O - 4*C) / D1
        expected["CH4"] = (H - 2*O + 4*C) / D2
        expected["CO2"] = (2*O + 4*C - H) / D2
        expected["N2"] = 2*N / D1

        expected["O2"] = 0
        expected["H2"] = 0
        expected["NH3"] = 0
    else:
        #Its unkown, prior should just be 1
        #So there is essentially no prior. Just want the likelihood
        return None
    return list(expected.values())

def calculatePrior(predAbun,sigmaPrior):
    '''
    predAbun: The predicted abundance for each moleecule
    sigmaPrior: The uncertainty for each molecule
    '''
    
    batchSize=predAbun.size(0)
    means=[]
    for i in range(batchSize):
        ev=calculateExpectedValues(predAbun[i].tolist())
        if ev is None:
            means.append(torch.zeros_like(predAbun[i]))
        else:
            means.append(torch.tensor(ev,device=predAbun.device))
    means=torch.stack(means,dim=0)


    normal = torch.distributions.Normal(0., 1.)
    # a = (0 - mu)/sigma,  b = (1 - mu)/sigma
    a = (0.0 - means) / sigmaPrior
    b = (1.0 - means) / sigmaPrior
    Z = normal.cdf(b) - normal.cdf(a)    # (B, M)

    # 3) unnormalized quadratic term
    quad = (predAbun - means)**2 / (2 * sigmaPrior**2)

    # 4) negative log of truncated PDF
    #    drop the +0.5*log(2πσ²) if you want simplicity
    nll_trunc = quad - torch.log(Z + 1e-12)  # add epsilon to avoid log(0)



    return nll_trunc.mean()



    var=sigmaPrior**2
    term1=((predAbun-means)**2) / (2*var)        
    return term1.mean()


def calculatePosterior(yReal,ySim,sigmaLikelihood,predAbun,sigmaPrior):
    # Math: P({y_{real}}|A_{pred}) \propto P(A_{pred}|Y_{real}) * P(A_{pred})
    '''
    This function calculates the unnormalized posterior

    Inputs
    ------
    yReal: Real, input transmittance data
    ySim: Simulated transmittance data, calculated using the predicted abundances
    sigmaLikelihood: Uncertainty of how well predicted abundances translate to simulated spectra
    predAbun: Predicted abundances
    sigmaPrior: Uncertaintiy of the precicted abundances

    Returns
    -------
    posterior: Unnormalized posterior probability
    '''


    prior=calculatePrior(predAbun,sigmaPrior) 
    likelihood=calculateLikelihood(yReal,ySim,sigmaLikelihood)
    nll_mean = torch.mean(likelihood)

    print(f"Likelihood: {nll_mean}")
    print(f"Prior: {prior}")

    posterior=nll_mean+prior
    return posterior


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.dataBased=nn.MSELoss()
        self.dataLossLambda=nn.Parameter(torch.zeros(1))
        self.detectionLambda=nn.Parameter(torch.zeros(1))
        self.posteriorLambda=nn.Parameter(torch.zeros(1))



    def forward(self,predAbun,uncertainty,realAbun,detectionOutput,config,inputTransmittance):#Predicted abundances, actual abundances, config file, real data
        '''
        predAbun: Predicted abundances
        uncertainty: Uncertainty about predicted abundances
        realAbun: True abundances
        detectionOutput: Output from detection model
        config: Config file
        inputTransmittance: Input spectral data
        '''
        

        #if detection says “present”
        #punish pred_abun<0.001
        #hinge_present = max(0, 0.001 − predAbun)
        hinge_present=F.relu(0.001 - predAbun)

        present_penalty=(detectionOutput*hinge_present).mean()

        #if detection says “absent”
        #punish pred_abun > 0.2
        #hinge_absent = max(0, pred_abun − 0.2)
        hinge_absent = F.relu(predAbun - 0.2)
        absent_penalty = ((1.0 - detectionOutput) * hinge_absent).mean()
        # 4) Combine
        detectionLoss = (80*present_penalty) + absent_penalty #This should make things work out better, and scale better


        dataLoss=self.dataBased(predAbun,realAbun)#Ranges from 0 to infinity


        
        
        sigmaLikelihood=0.1#How well simulated data matches predicted abundances. How accurate simulations are

        
        # simulatedTransmittance=testNewAbundances(config,predAbun)
        # simulatedTransmittance=simulatedTransmittance.to(device)


        # #Need to figure out sigmaLikelihood and muPrior
        # posterior=calculatePosterior(inputTransmittance,simulatedTransmittance,sigmaLikelihood,predAbun,uncertainty)
        

        posterior=torch.tensor(0.3257)
        #Combine the loss
        #See Kendall et al. “Multi‐Task Learning Using Uncertainty to Weigh Losses”
        totalLoss=(
            posterior*torch.exp(-self.posteriorLambda) + self.posteriorLambda
            + dataLoss*torch.exp(-self.dataLossLambda) + self.dataLossLambda
            + detectionLoss*torch.exp(-self.detectionLambda) + self.detectionLambda


        )
        #For now, just add everything, but the in future, can play with the idea of multiplying each by some factor 
        return totalLoss
    
#Lets use "C:\Users\Tristan\Downloads\HyPCAR3\data\A\A1_1.csv"
#Lets look at config: C:\Users\Tristan\Downloads\HyPCAR3\configFiles\A1_1.txt
#0.0, 0.0, 0.04146731671062273, 0.0, 0.45995175433673047, 0.1069055790034298, 0.39167534994921704
#Okay, to use this, copy the config file into tempConfig, then run this. YOu will get a new config file. 
#Take this config file and get the results with your PSG
#Put that info into tempData.csv

moleculeDetection=detectionModel()
abundanceDetection=abundanceModel()


moleculeDetection.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\flexibleDetectionModel.pt",weights_only=True))
abundanceDetection.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\HyPCAR3\finalBaseAbundance.pt",weights_only=True))

moleculeDetection.eval()
abundanceDetection.eval()

data=pd.read_csv(r"C:\Users\Tristan\Downloads\HyPCAR3\data\A\A1_1.csv")

wavelength,transmittance=list(map(wavelengthFilter,map(str,data.iloc[:,0]))),data.iloc[:,1]
input_data=torch.tensor(np.stack([wavelength, transmittance], axis=1), dtype=torch.float32)

# #add a batch dimension (1, since it's one example)
input_data=input_data.unsqueeze(0)



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


with torch.no_grad():
    detectionOutput=moleculeDetection(input_data)
    abundanceOutput, uncertainty, attentionWeights = abundanceDetection(input_data, detectionOutput)



# Load the “true” simulated spectrum
simData = pd.read_csv(r"C:\Users\Tristan\Downloads\HyPCAR3\physcicsFineTune\tempData.csv")
w, t = simData.iloc[:,0].values.astype(float), simData.iloc[:,1].values.astype(float)

# Plot raw spectra: input vs simulated
plt.figure(figsize=(6,3))
plt.plot(wavelength, transmittance, color="blue", label="Input (y_real)")
plt.plot(w, t,            color="orange", label="Simulated (y_sim)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Transmittance")
plt.legend()
plt.tight_layout()
plt.show()

# Compute your loss terms
lik_vals   = calculateLikelihood(input_data, torch.tensor(np.stack([w,t],axis=1), dtype=torch.float32), 0.1)
prior_val  = calculatePrior(abundanceOutput, uncertainty)
post_val   = calculatePosterior(input_data, torch.tensor(np.stack([w,t],axis=1), dtype=torch.float32),
                                0.1, abundanceOutput, uncertainty)
print("NLL per sample:", lik_vals)
print("Prior loss:",      prior_val)
print("Posterior loss:",  post_val)

# ── NEW: Gaussian Likelihood + Prior Curves ──
molecules = ["O2","N2","H2","CO2","H2O","CH4","NH3"]

# 1) Gaussian Likelihood plot
x_res = np.linspace(-1, 1, 500)
sigma_lik = 0.1  # your chosen likelihood sigma
lik_pdf = norm.pdf(x_res, loc=0, scale=sigma_lik)

plt.figure(figsize=(5, 3))
plt.plot(x_res, lik_pdf, color='C0', lw=2)
plt.title("Gaussian Likelihood Distribution", fontsize=12)
plt.xlabel("Residual ΔA", fontsize=10)
plt.ylabel("Probability Density", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

x_abun = np.linspace(0, 1, 500)

plt.figure(figsize=(6, 4))
for i, mol in enumerate(molecules):
    mu       = abundanceOutput[0, i].item()
    sigma_pr = uncertainty[0, i].item()
    
    # compute standard (unnormalized) Gaussian PDF on [0,1]
    pdf_unnorm = norm.pdf(x_abun, loc=mu, scale=sigma_pr)
    
    # compute the truncation constant Z = Phi((1-mu)/sigma) - Phi((-mu)/sigma)
    a, b = (0 - mu) / sigma_pr, (1 - mu) / sigma_pr
    Z = norm.cdf(b) - norm.cdf(a)
    
    # truncated PDF
    prior_pdf = pdf_unnorm / (Z + 1e-12)  # small eps to avoid div0
    
    plt.plot(x_abun, prior_pdf, lw=1.7, label=mol)

plt.title("Truncated-Normal Priors for All Molecules", fontsize=14)
plt.xlabel("Predicted Abundance A", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend(fontsize=8, ncol=2, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()


plt.show(block=True)




real=[0.0 , 0.0 , 0.2772344370097009, 0.0 , 0.13771633168201375, 0.4698620331291393, 0.11518719817914612]
#[0.040885258466005325, 0.06607979536056519, 0.3177936375141144, 5.715466855349405e-08, 0.2154405266046524, 0.1535840481519699, 0.20621664822101593]
#0,0,1,0,1,1,1
molecules = ["O2","N2","H2","CO2","H2O","CH4","NH3"]
pred = np.array(abundanceOutput.tolist()[0])
print(detectionOutput)

true = np.array(real)
sigma = np.array(uncertainty.tolist()[0])
size = (sigma / sigma.max()) * 200 + 50  # range ~50–250

# Reduce overall figure size (graph becomes smaller)
plt.figure(figsize=(7,7))
ax = plt.gca()

# 1) Identity line
lims = [0, 1]
ax.plot(lims, lims, '--', color='gray', linewidth=1)

# 2) Scatter, sizing by uncertainty
sc = ax.scatter(true, pred,
                s=size,
                c=sigma,
                cmap='viridis',
                edgecolor='k',
                alpha=0.8)

# 3) Annotate each point with molecule name
for i, mol in enumerate(molecules):
    ax.text(true[i], pred[i],
            mol, fontsize=12,
            va='bottom', ha='right')

# 4) Colorbar for uncertainty with a smaller size using shrink
cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.7)
cbar.set_label("Predicted σ", rotation=270, labelpad=15, fontsize=16)

# Styling
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("True Abundance", fontsize=16)
ax.set_ylabel("Predicted Abundance", fontsize=16)
ax.set_title("Predicted vs True Abundances with Uncertainty Encoding", fontsize=16, pad=20)
ax.grid(alpha=0.3)
ax.set_aspect('equal', 'box')

plt.tight_layout()

plt.show()


hinge_present=F.relu(0.001 - torch.tensor(pred))

present_penalty=(detectionOutput[0]*hinge_present)



hinge_absent = F.relu(torch.tensor(pred) - 0.2)
absent_penalty = ((1.0 - detectionOutput[0]) * hinge_absent)

present_penalty=np.array((present_penalty*80).tolist())
absent_penalty=np.array(absent_penalty.tolist())

N = len(molecules)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Append first value to end for each series
temp=list(present_penalty)
present=np.array(temp+[temp[0]])

temp=list(absent_penalty)
absent=np.array(temp+[temp[0]])


print(present)
print(absent)
print("haiosd")
print(present_penalty)
print(absent_penalty)
# Create radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Plot data
ax.plot(angles, present, label="Present Penalty ×80", color='C0', linewidth=2)
ax.fill(angles, present, color='C0', alpha=0.25)

ax.plot(angles, absent, label="Absent Penalty", color='C1', linewidth=2)
ax.fill(angles, absent, color='C1', alpha=0.25)

# Format the chart
ax.set_theta_offset(np.pi / 2)         # start from top
ax.set_theta_direction(-1)             # clockwise
ax.set_thetagrids(np.degrees(angles[:-1]), molecules,fontsize=16)
ax.set_title("Detection Consistency Penalties per Molecule", y=1.1,fontsize=20)
ax.set_rlabel_position(180 / N)        # radial labels aligned
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1),fontsize=14)

plt.tight_layout()
plt.show()

# with torch.no_grad():
#     detectionOutput=moleculeDetection(input_data)
#     abundanceOutput,uncertainty,attentionWeights=abundanceDetection(input_data,detectionOutput)


# print(detectionOutput)
# print(abundanceOutput.tolist())

testnewAbundance(abundanceOutput.tolist()[0])
# print(uncertainty.tolist())

# simData=pd.read_csv(r"C:\Users\Tristan\Downloads\HyPCAR3\physcicsFineTune\tempData.csv")
# w,t=list(map(float,simData.iloc[:,0])),list(map(float,simData.iloc[:,1]))
# print(w)
# print(t)
# simData=torch.tensor(np.stack([w, t], axis=1), dtype=torch.float32)
# plt.plot(wavelength,transmittance,color="blue")
# plt.plot(w,t,color="orange")
# plt.show(block=True)
# print(calculateLikelihood(simData,input_data,0.1))
# print(calculatePrior(abundanceOutput,uncertainty))
# print(calculatePosterior(input_data,simData,0.1,abundanceOutput,uncertainty))

# criterion = customLoss()
# for p in criterion.parameters():
#     assert p.requires_grad

# # # dummy batch
# # B, C, L = 2, 1, 10
# # y_real  = torch.randn(B, C, L, requires_grad=True)
# # y_sim   = torch.randn(B, C, L, requires_grad=True)
# # pred_abun     = torch.randn(B, 7, requires_grad=True).abs()
# # sigma_prior   = torch.full((B,7), 0.1, requires_grad=True)
# # real_abun     = torch.randn(B, 7, requires_grad=True).abs()
# # detection_out = torch.rand(B, 7, requires_grad=True)

# loss = criterion(abundanceOutput, uncertainty, torch.tensor([[0.0 , 0.0 , 0.2772344370097009, 0.0 , 0.13771633168201375, 0.4698620331291393, 0.11518719817914612]]), detectionOutput, None, simData)
# loss.backward()

# print([p.grad.norm().item() for p in criterion.parameters()])  # should all be >0
