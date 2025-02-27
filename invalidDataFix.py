import os
import pandas as pd
import torch
import pypsg
import csv
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

#Need to remake tar files, after updating data too
atmosphereTypes=["A","B","C","None"]
folderPath="/home/tristanb/scratch/data"
for aType in atmosphereTypes:
    curFolderPath=folderPath+"/"+aType

    for path in os.lisdir(curFolderPath):
        data=pd.read_csv(os.path.join(curFolderPath,path))

        wavelength=list(map(wavelengthFilter,data.iloc[:,0]))#Removes um from wavelength data
        transmittance=list(data.iloc[:,1])

        combinedData=torch.tensor(list(zip(wavelength, transmittance)), dtype=torch.float32)

        #nan is present OR data genereated weird
        if torch.isnan(combinedData).any() or len(wavelength)<784:
            configFilePath="/home/tristanb/scratch/configFiles/"

            fileName=os.path.basename(path)
            fileName=fileName.removesuffix(".csv")
            configFilePath+=fileName+".txt"

            cfg=pypsg.Pyconfig.from_file(fileName)
            psg=pypsg.APICall(cfg=cfg,output_type="trn")
            response=psg()
            data=response.trn
            wavelength,transmittance=data["Wave/freq"],data["Total"]

            #The new file path for the newly generated data will be dataFile
            with open(os.path.join(curFolderPath,path),mode="w",newline="") as f:
                writer = csv.writer(f)
                for i in range(len(wavelength)):
                    writer.writerow((wavelength[i],transmittance[i]))










