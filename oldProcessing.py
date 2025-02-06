import dask
from dask.distributed import Client
import pypsg
import csv
import logging
import os
from retrying import retry

@retry(stop_max_attempt_number=5, wait_fixed=200)#Retry 3 times, with 200 miiliseconds inebtween
def generate_spectra(cfg_path):
    '''
    Calls PSG API.

    Inputs
    ------
    cfg_path: The file path

    Returns
    -------
    response.trn: The transmittance data
    '''
    try:
        cfg = pypsg.cfg.PyConfig.from_file(cfg_path)
        psg = pypsg.APICall(cfg=cfg, output_type='trn')
        response = psg()
        return response.trn
    except Exception as e:
        logging.error(f"Error processing {cfg_path}: {e}")
        raise
def createDataFile(data,filePath,folderPath):
    '''
    This function will create the data file containing the wavelength, and transmittance data
    It will create this file in the right folder.

    Inputs
    ------
    data: The raw spectral data
    filePath: The file path for the config file. The data file will have the same path, just stored somewhere else
    folderPath: The folder path where this data file will go.

    Returns
    -------
    None
    '''
    wavelength,transmittance=data["Wave/freq"],data["Total"]
    #C:\Users\Tristan\Downloads\HyPCAR\configFiles\O2-1.txt
    prefix=r"C:\Users\Tristan\Downloads\HyPCAR\configFiles"
    name=filePath.removeprefix(prefix)
    name=name[1:]#Remove backslash
    name=name.removesuffix(".txt")

    newFileName=folderPath+f"\\{name}"+".csv"
    # print(f"HERE: {newFileName}")
    with open(newFileName,mode="w",newline="") as f:
        writer = csv.writer(f)
        for i in range(len(wavelength)):
            writer.writerow((wavelength[i],transmittance[i]))
        
def callPSG(configs, moleculeCombination):
    '''
    This function will take all the config files and pass them through PSG to get transmittance data.

    Inputs
    -----
    configs: The list of file paths for configuration paths
    moleculeCombination: The molecule combinations

    Returns
    -------
    None, the files will be created    
    '''
    folderPath = os.path.join("C:\\Users\\Tristan\\Downloads\\HyPCAR\\data", "-".join(moleculeCombination))
    os.makedirs(folderPath, exist_ok=True)
    print(f"Folder created: {folderPath}")
    client=Client(n_workers=4)
    batch_size = 4

    for i in range(0, len(configs), batch_size):
        batchPaths = configs[i:i + batch_size]
        futures = [(cfg_path, dask.delayed(generate_spectra)(cfg_path)) for cfg_path in batchPaths]#Fixed this so that it now corresponds the file with result
        results = dask.compute(*[f[1] for f in futures])
        for j, result in enumerate(results):
            if result is not None:
                createDataFile(result, futures[j][0], folderPath)
    client.close()