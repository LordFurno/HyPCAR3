import os
from contextlib import contextmanager

@contextmanager
def change_dir(new_dir):
    original_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(original_dir)

def generate_spectra(config_path):
    # Delay the import here so that each child loads pypsg afresh.
    import pypsg

    abs_config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(abs_config_path)
    
    with change_dir(config_dir):
        cfg = pypsg.cfg.PyConfig.from_file(abs_config_path)
        psg = pypsg.APICall(cfg=cfg, output_type='trn')
        response = psg()
        return response.trn

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    config_paths=[r"C:\Users\Tristan\Downloads\HyPCAR\workingDirectory\working-"+f"{i}.txt" for i in range(32)]
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(generate_spectra, config_paths))
    
    for idx, spectra in enumerate(results):
        print(f"Spectra {idx+1}: {spectra}")
