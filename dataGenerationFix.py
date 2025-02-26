import os
import pypsg
import csv

#To fix generate anything that wasn't generated
"/home/tristanb/scratch/configFiles"
configFolder="/home/tristanb/scratch/configFiles"
for fileName in os.listdir(configFolder):
    atmosphereType,numbers=fileName.split("_")
    if len(atmosphereType)==2:
        #A1 or A2 
        atmosphereType="A"
    dataFolder="/home/tristanb/scratch/data/"+atmosphereType

    dataFile=dataFolder+"/"+fileName
    dataFile=dataFile.removesuffix(".txt")
    dataFile+=".csv"
    
    if not os.path.isfile(dataFile):
        #This file was not genereated
        #Send to pypsg
        cfg=pypsg.Pyconfig.from_file(fileName)
        psg=pypsg.APICall(cfg=cfg,output_type="trn")
        response=psg()
        data=response.trn
        wavelength,transmittance=data["Wave/freq"],data["Total"]

        #The new file path for the newly generated data will be dataFile
        with open(dataFile,mode="w",newline="") as f:
            writer = csv.writer(f)
            for i in range(len(wavelength)):
                writer.writerow((wavelength[i],transmittance[i]))



