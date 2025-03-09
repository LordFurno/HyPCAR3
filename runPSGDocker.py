import asyncio
import torch
import numpy as np
from io import StringIO

# Asynchronous function to execute the command
async def execute_command_async(file_path):
    command = [
        "docker", "exec", "-t", "psgWithDirectoryPort8080",
        "curl", "-d", "type=trn", "--data-urlencode", f"file@/containerDirectory/{file_path}",
        "http://host.docker.internal:8080/api.php"
    ]
    
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


async def run_commands_async(file_paths):
    tasks=[execute_command_async(file_path) for file_path in file_paths]
    results=await asyncio.gather(*tasks)  
    return results

def get_data_async():
    #Not matching files properly, :)
    file_paths=[f"working-{i}.txt" for i in range(32)]
    #Run the asynchronous tasks and gather the results
    results=asyncio.run(run_commands_async(file_paths))

    #Extract data
    # data = []
    # for result in results:
    #     if result[1]:  
    #         try:
    #             data_array = np.loadtxt(StringIO(result[1]), comments='#')
    #             # Extract columns 0 and 1 as a list of tuples
    #             data_array = list(zip(data_array[:, 0], data_array[:, 1]))[:-1]
    #             # Convert to PyTorch tensor
    #             data.append((result[0], torch.tensor(data_array)))
    #         except Exception as e:
    #             print(f"Error processing data for {result[0]}: {e}")

 
    data=[]
    for result in results:
        if result[1]:  
            try:
                data_array=np.loadtxt(StringIO(result[1]), comments='#')
                # Extract columns 0 and 1 as a list of tuples
                data_array=list(zip(data_array[:, 0], data_array[:, 1]))[:-1]
                # Convert to PyTorch tensor
                data.append(torch.tensor(data_array))
            except Exception as e:
                print(f"Error processing data for {result[0]}: {e}")

    # Stack the tensors into one tensor
    if data:
        data=torch.stack(data)
        return data
    else:
        return torch.tensor([])  # Return an empty tensor if no data is processed

if __name__ == "__main__":
    import time
    start=time.time()
    data=get_data_async()
    print(data)
    print(data.size())
    print(time.time() - start)
#13.182 seconds for 32
