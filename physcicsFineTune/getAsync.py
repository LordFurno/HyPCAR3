import asyncio
import numpy as np
import torch
from io import StringIO

# Asynchronous function to execute the command with the provided index
async def execute_command_async(file_path, index):
    command = [
        "curl", "-d", "type=trn", "--data-urlencode", f"file@{file_path}",
        "http://localhost:8080/api.php"
    ]
    try:
        # Run the subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return index, stdout.decode()
        else:
            print(f"Error executing command for {file_path}: {stderr.decode()}")
            return index, ""
    except Exception as e:
        print(f"Exception occurred for {file_path}: {e}")
        return index, ""

async def run_commands_async(file_paths):
    tasks = [execute_command_async(file_path, i) for i, file_path in enumerate(file_paths)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda x: x[0])

def get_data_async():
    file_paths = [f"workingDirectory/working-{i}.txt" for i in range(32)]

    results = asyncio.run(run_commands_async(file_paths))

    data = [None] * len(file_paths)

    for index, output in results:
        if output:
            try:
                data_array = np.loadtxt(StringIO(output), comments='#')
                # Ensure data is 2D
                if data_array.ndim == 1:
                    data_array = np.expand_dims(data_array, axis=0)
                # Extract columns 0 and 1 and skip last row
                processed_data = data_array[:-1, :2]
                data[index] = torch.tensor(processed_data, dtype=torch.float32)
            except Exception as e:
                print(f"Error processing data for file index {index}: {e}")
                data[index] = None
        else:
            data[index] = None

    valid_data = [d for d in data if d is not None]

    if valid_data:
        try:
            stacked_data = torch.stack(valid_data)
            return stacked_data
        except Exception as e:
            print(f"Error stacking tensors: {e}")
            return torch.empty(0)
    else:
        return torch.empty(0)

# # Run
# if __name__ == "__main__":
#     import time
#     start = time.time()
#     tensor_data = get_data_async()
#     print(tensor_data.shape)
#     print(f"Execution time: {time.time() - start:.2f} seconds")
