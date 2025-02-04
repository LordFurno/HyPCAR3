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


