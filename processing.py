import concurrent.futures
import logging
import pypsg
import time
import os
import csv

from retrying import retry

@retry(stop_max_attempt_number=5, wait_fixed=1)#Retry 3 times, with 200 miiliseconds inebtween
def generate_spectra(configPath):
    abs_config_path = os.path.abspath(configPath)
    os.chdir(os.path.dirname(abs_config_path))

    
    cfg = pypsg.cfg.PyConfig.from_file(abs_config_path)
    psg = pypsg.APICall(cfg=cfg, output_type='trn')
    response = psg()
    return response.trn


def createDataFile(data,filePath,folderPath):
    wavelength,transmittance=data["Wave/freq"],data["Total"]

    prefix=r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles"
    
    name=filePath.removeprefix(prefix)
    name=name[1:]#Remove backslash
    name=name.removesuffix(".txt")

    newFileName=folderPath+f"\\{name}"+".csv"



    with open(newFileName,mode="w",newline="") as f:
        writer = csv.writer(f)
        for i in range(len(wavelength)):
            writer.writerow((wavelength[i],transmittance[i]))


def callPSG(configs,atmosphereType):
    folderPath = os.path.join("C:\\Users\\Tristan\\Downloads\\HyPCAR3\\data", atmosphereType)
    os.makedirs(folderPath, exist_ok=True)
    results=[]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futureToConfigs = {executor.submit(generate_spectra, cp): cp for cp in configs}

        for future in concurrent.futures.as_completed(futureToConfigs):
            cp=futureToConfigs[future]
            try:
                result=future.result()

                createDataFile(result,cp,folderPath)

                results.append(result)

            except Exception as exc:
                logging.error(f"{cp} generated an exception: {exc}")
    # print(len(results))

if __name__ == "__main__":
    start=time.time()
    config_paths = [r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles\O2-N2-CO2-H2O-N2O-CH4-H2S-724.txt" for i in range(16)]
    callPSG(config_paths,"A")
    print(time.time()-start)
