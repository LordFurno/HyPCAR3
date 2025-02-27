#!/bin/bash
#SBATCH --account=def-pjmann
#SBATCH --job-name=psgDataGeneration
#SBATCH --output=psgDataGeneration-%J.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1   
#SBATCH --mem=6G
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=sirfurno@gmail.com
#SBATCH --mail-type=ALL



module load python/3.12
module load mpi4py/3.1.6

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r /home/tristanb/projects/def-pjmann/tristanb/dataGenerationRequirements.txt
#55667231

cd /home/tristanb/projects/def-pjmann/tristanb

python dataGeneration.py


