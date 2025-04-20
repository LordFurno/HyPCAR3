import asyncio
import numpy as np
from io import StringIO
import torch
import os
# Asynchronous function to execute the command
async def execute_command_async(file_path):
    command = [
        "curl", "-d", "type=trn", "--data-urlencode", f"file@{file_path}",
        "http://localhost:4000/api.php"
    ]
    #curl -d type=trn --data-urlencode file@configTemplate.txt http://localhost:8080/api.php
    try:
        # Run the subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout,stderr=await process.communicate()
        
        if process.returncode==0:
            return file_path, stdout.decode()
        else:
            print(f"Error executing command for {file_path}: {stderr.decode()}")
            return file_path,stderr.decode()
    except Exception as e:
        print(f"Exception occurred for {file_path}: {e}")
        return file_path, ""





#3399153
async def run_commands_async(file_paths):
    tasks=[execute_command_async(file_path) for file_path in file_paths]
    results=await asyncio.gather(*tasks)  
    return results

def get_data_async():

    file_paths=[r"C:\Users\Tristan\Downloads\HyPCAR3\configTemplate.txt" for i in range(32)]
    
    #Run the asynchronous tasks and gather the results
    results=asyncio.run(run_commands_async(file_paths))

    data = []
    for result in results:
        if result[1]:  
            try:
                data_array = np.loadtxt(StringIO(result[1]), comments='#')
                # Extract columns 0 and 1 as a list of tuples
                data_array = list(zip(data_array[:, 0], data_array[:, 1]))[:-1]
                # Convert to PyTorch tensor
                temp=os.path.basename(result[0])
                num=temp.removeprefix("working-")
                num=num.removesuffix(".txt")


                data.append((int(num), torch.tensor(data_array)))
            except Exception as e:
                print(f"Error processing data for {result[0]}: {e}")
    data.sort(key=lambda x:x[0])
    data=[d[1] for d in data]


    # Stack the tensors into one tensor
    if data:
        data=torch.stack(data)
        return data
    else:
        return torch.tensor([])  # Return an empty tensor if no data is processed

if __name__ == "__main__":
    import time
    start=time.time()
    print(get_data_async().shape)
    print(time.time() - start)
#13.182 seconds for 32
