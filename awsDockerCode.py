#!/usr/bin/env python3
import asyncio
import numpy as np
from io import StringIO
import aiohttp
import urllib.parse
import time
API_URL = "http://127.0.0.1:8080/api.php"


# podman run --userns=keep-id -d --name psg -p 127.0.0.1:8080:8080 --volume ~/my-var-run:/var/run --volume ~/projects/def-pjmann/tristanb/tempConf.conf:/etc/httpd/conf/httpd.conf --volume ~/projects/def-pjmann/tristanb/tempwww.conf:/etc/php-fpm.d/www.conf psgimage
# podman run --userns=keep-id -d --name psg -p 127.0.0.1:8080:8080 --volume ~/my-var-run:/var/run:rw --volume ~/projects/def-pjmann/tristanb/tempConf.conf:/etc/httpd/conf/httpd.conf --volume ~/projects/def-pjmann/tristanb/tempwww.conf:/etc/php-fpm.d/www.conf psgimage

# apptainer instance start \ 
#     -B ~/my-var-run:/var/run:rw \ 
#     -B ~/projects/def-pjmann/tristanb/tempConf.conf:/etc/httpd/conf/httpd.conf \
#     -B ~/projects/def-pjmann/tristanb/tempwww.conf:/etc/php-fpm.d/www.conf \ 
#     -B ~/apache_logs:/etc/httpd/logs \ 
#     psgimage.sif psg


# apptainer instance start \
#   -B ~/my-var-run:/var/run:rw \
#   -B ~/projects/def-pjmann/tristanb/tempConf.conf:/etc/httpd/conf/httpd.conf \
#   -B ~/projects/def-pjmann/tristanb/tempwww.conf:/etc/php-fpm.d/www.conf \
#   -B ~/apache_logs:/etc/httpd/logs \
#   psgimage.sif psg


# apptainer instance start \
#   -B ~/my-var-run:/var/run:rw \
#   -B ~/projects/def-pjmann/tristanb/tempConf.conf:/etc/httpd/conf/httpd.conf:rw \
#   -B ~/projects/def-pjmann/tristanb/tempwww.conf:/etc/php-fpm.d/www.conf:rw \
#   -B ~/apache_logs:/etc/httpd/logs:rw \
#   -B ~/phpfpm_logs:/var/log/php-fpm:rw \
#   -B ~/projects/def-pjmann/tristanb/tempResults:/var/www/html/results:rw \
#   workingPSG.sif psg

# apptainer exec instance://psg /usr/sbin/httpd -D FOREGROUND &
# apptainer exec instance://psg /usr/sbin/php-fpm


# apptainer instance start \
#   -B ~/my-var-run:/var/run \
#   -B ~/projects/def-pjmann/tristanb/tempConf.conf:/etc/httpd/conf/httpd.conf \
#   -B ~/projects/def-pjmann/tristanb/tempwww.conf:/etc/php-fpm.d/www.conf \
#   -B ~/apache_logs:/etc/httpd/logs \
#   psgimage.sif psg

# apptainer exec instance://psg /usr/sbin/httpd -D FOREGROUND &

async def fetch_and_parse(session, file_path):
    # 1) Read the entire file as text
    with open(file_path, 'r') as f:
        file_content = f.read()

    # 2) Build the payload exactly like `curl -d type=trn --data-urlencode file@...`
    payload = {
        "type": "trn",
        "file": file_content
    }
    encoded = urllib.parse.urlencode(payload)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # 3) Post and await the text response
    async with session.post(API_URL, data=encoded, headers=headers) as resp:
        text = await resp.text()

    # 4) Debug: print what came back (uncomment if needed)
    # print(f"\n--- Response for {file_path} ---\n{text}\n------------------------------\n")

    if not text.strip():
        print(f"⚠️ No data returned for {file_path}")
        return None

    # 5) Parse into a NumPy array of (col0, col1) pairs
    arr = np.loadtxt(StringIO(text), comments='#')
    # If the server sometimes returns a single column or empty, guard against it:
    if arr.ndim != 2 or arr.shape[1] < 2:
        print(f"⚠️ Unexpected shape {arr.shape} from {file_path}")
        return None

    pairs = list(zip(arr[:, 0], arr[:, 1]))[:-1]
    return np.array(pairs)

async def get_data_async(file_paths):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_parse(session, fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)

    # Filter out any failures
    data = [r for r in results if r is not None]
    if not data:
        return np.array([])

    # Stack into a single array of shape (n_files, n_rows, 2)
    return np.stack(data)

def main():
    # duplicate your single config twice (or list multiple distinct files)
    file_paths = ["configTemplate.txt" for i in range(3)]
    data = asyncio.run(get_data_async(file_paths))
    print("Final stacked data shape:", data.shape)
    # e.g. (2, N, 2)
    # And if you want to inspect:
    # print(data)

if __name__ == "__main__":
    start=time.time()
    main()
    print(time.time()-start)