import docker
import time

client = docker.from_env()

hostDirectory = r"C:\Users\Tristan\Downloads\HyPCAR\workingDirectory"
containerDirectory = "/containerDirectory"
volumes = {
    hostDirectory: {
        "bind": containerDirectory,
        "mode": "ro"   # Use "rw" if needed
    }
}

# Run the container in the background.
container = client.containers.run(
    "psg",         # Your image name
    detach=True,
    tty=True,
    stdin_open=True,
    volumes=volumes
)
# Combine commands into one shell script.
# This launches all curl calls in the background and waits for them.
cmd = r"""bash -c '
for i in {0..31}; do
    curl -d "type=trn" --data-urlencode "file@/containerDirectory/working-${i}.txt" "http://host.docker.internal:8080/api.php" &
done
wait'"""

start = time.time()
result = container.exec_run(cmd)
end = time.time()

print(result.output.decode())
print("Elapsed time:", end - start)

container.stop()
container.remove()
