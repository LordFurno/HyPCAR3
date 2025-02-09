import pypsg
import logging
# import matplotlib.pyplot as plt
import dask
from dask.distributed import Client
def generate_spectra(configPath):
    try:
        cfg=pypsg.cfg.PyConfig.from_file(configPath)
        psg=pypsg.APICall(cfg=cfg, output_type='trn')
        response=psg()
        return response.trn
    except Exception as e:
        logging.error(f"Error processing {configPath}: {e}")
        raise

import time
start=time.time()
for i in range(32):
    print(generate_spectra(r"C:\Users\Tristan\Downloads\HyPCAR3\configFiles\O2-N2-CO2-H2O-N2O-CH4-H2S-724.txt"))
print(time.time()-start)
