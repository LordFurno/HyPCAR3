
def testNewAbundances(fileName,moleculeAbundances):
    '''
    This function will update a temporary config file with the predictions from the ml model. 
    This is for the PSG-based loss, it takes in a batch of 32 different files and abundances. If this

    Inputs
    -----
    fileNames: The names of the config files
    moleculeAbundances: The new predicted abundances from the model.

    Returns
    -------
    combinedData: A tensor that contains the new wavelength and transmittance values for the updated abundances
    '''
    #Fix this functino, doesn't work because stupid batches. really really annoying, can think about using dask
    #To speed up the process because I'm dealing with about 32 files.
    #Super annoying, will create something called a "working directory" where I will store the modified config files per batch
    #Then use batch to run data through them, will need to pretty much rewrite everything here

    lines=[]
    with open(fileName) as f:
        for line in f:
            lines.append(line)
        
    tempFile=r"C:\Users\Tristan\Downloads\HyPCAR3\temp.txt"
    #Deal with atmosphere layers here
    #Start line is 54

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

    with open(tempFile,"w") as f:
        f.writelines(lines)
def testNewAbundancesWithHe(fileName,moleculeAbundances,tempFile=None):
    '''
    This function will update a temporary config file with the predictions from the ml model. 
    This is for the PSG-based loss, it takes in a batch of 32 different files and abundances. If this

    Inputs
    -----
    fileNames: The names of the config files
    moleculeAbundances: The new predicted abundances from the model.

    Returns
    -------
    combinedData: A tensor that contains the new wavelength and transmittance values for the updated abundances
    '''
    #Fix this functino, doesn't work because stupid batches. really really annoying, can think about using dask
    #To speed up the process because I'm dealing with about 32 files.
    #Super annoying, will create something called a "working directory" where I will store the modified config files per batch
    #Then use batch to run data through them, will need to pretty much rewrite everything here

    lines=[]
    with open(fileName) as f:
        for line in f:
            lines.append(line)
        
    if tempFile==None:
        tempFile=r"C:\Users\Tristan\Downloads\HyPCAR3\temp.txt"
    else:
        pass
    #Deal with atmosphere layers here
    #Start line is 54

    abundanceDictionary={}
    molecules=["O2","N2","H2","CO2","H2O","CH4","NH3","He"]
    for i in range(len(moleculeAbundances)):
        abundanceDictionary[molecules[i]]=moleculeAbundances[i]

    moleculeWeights={"O2":31.999, "N2":28.02, "H2":2.016,"CO2":44.01, "H2O":18.01528,"CH4":16.04,"NH3":17.03052,"He":4}#g/mol
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
    
    nMolecules=8

    HITRANValues={"O2":"HIT[7]","N2":"HIT[22]","H2":"HIT[45]","CO2":"HIT[2]","H2O":"HIT[1]","CH4":"HIT[6]","NH3":"HIT[11]","He":"HIT[0]0"}

    #Additional parameters to actually run the data properly.
    lines[42]="<ATMOSPHERE-NGAS>"+str(len(moleculeAbundances))+"\n" #Number of gases are in the atmosphere
    lines[43]="<ATMOSPHERE-GAS>"+",".join(molecules)+"\n" #What gases are in the atmosphere
    lines[44]="<ATMOSPHERE-TYPE>"+",".join(HITRANValues[mol] for mol in molecules)+"\n" #HITRAN values for each gas
    lines[45]="<ATMOSPHERE-ABUN>"+"1,"*(len(moleculeAbundances)-1)+"1"+"\n" #Molecule abunadnces. They're all 1, because abundances are defined in vertical profile
    lines[46]="<ATMOSPHERE-UNIT>"+"scl,"*(len(moleculeAbundances)-1)+"scl"+"\n" #Abundance unit
    lines[49]="<ATMOSPHERE-WEIGHT>"+str(averageWeight)+"\n" #Molecule weight of atmosphere g/mol
    lines[52]="<ATMOSPHERE-LAYERS-MOLECULES>"+",".join(molecules)+"\n" #Molecule in vertical profile

    with open(tempFile,"w") as f:
        f.writelines(lines)

#Taurex results:
# testNewAbundancesWithHe(r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles\B_8104.txt",[0,0.4963,0.0959,0.0929,0.0481,0.097,0.0734,0.0987])

#HyPCAR results:
testNewAbundances(r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles\B_8104.txt",[0.1154,0.5474,0.0,0.1856,0.1517,0.0,0.0])



#Eartg testing
# testNewAbundances(r"C:\Users\Tristan\Downloads\HyPCAR3\earthConfigTemplate.txt",[0.8121215105056763, 0.022806989029049873, 0.0025340875145047903, 1.3289949492900632e-05, 0.01999226026237011, 0.13940148055553436, 0.003130426397547126])

# testNewAbundances(r"C:\Users\Tristan\Downloads\HyPCAR3\lhs1140b.txt",[0.03443722054362297, 0.18520966172218323, 0.08198410272598267, 0.016101757064461708, 0.27920645475387573, 0.19176138937473297, 0.21129940450191498])