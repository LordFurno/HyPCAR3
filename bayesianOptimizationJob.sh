#!/bin/bash
#SBATCH --account=def-pjmann
#SBATCH --job-name=bayesianOptimization
#SBATCH --output=bayesianOptimization-%J.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=sirfurno@gmail.com
#SBATCH --mail-type=ALL

#Loading modules
module load python/3.12
module load mpi4py/3.1.6
module load gcc arrow

export HEAD_NODE=$(hostname) #store head node's address
export RAY_PORT=34567 #choose a port to start Ray on the head node 


virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r /home/tristanb/projects/def-pjmann/tristanb/pureCNNRequirements.txt
pip install ray --no-index
pip install "ray[tune]"
pip install optuna

mkdir $SLURM_TMPDIR/data
tar xf ~/data.tar -C $SLURM_TMPDIR/data --strip-components=4
#strip components is so it doesn't create a ton of extra directories
#Do the above for the config files when training the abundance model
mkdir $SLURM_TMPDIR/configFiles
tar xf ~/configFiles.tar -C $SLURM_TMPDIR/configFiles --strip-components=4

ray start --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=4 --block &
sleep 10


cd /home/tristanb/projects/def-pjmann/tristanb
python bayesianOptimization.py
