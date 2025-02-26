#!/bin/bash
#SBATCH --account=def-pjmann
#SBATCH --job-name=detectionModelTraining
#SBATCH --output=detectionModelTraining-%J.out
#SBATCH --time=
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=sirfurno@gmail.com
#SBATCH --mail-type=ALL



module load python/3.12 cuda cudnn


virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r /home/tristanb/projects/def-pjmann/tristanb/detectionModelRequirements.txt

mkdir $SLURM_TMPDIR/data
tar xf ~/projects/def-pjmann/tristanb/data.tar -C $SLURM_TMPDIR/data
#Do the above for the config files when training the abundance model

cd /home/tristanb/projects/def-pjmann/tristanb

python detectionModel.py


