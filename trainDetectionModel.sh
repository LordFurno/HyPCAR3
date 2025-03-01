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
tar xf ~/data.tar -C $SLURM_TMPDIR/data --strip-components=4

mkdir $SLURM_TMPDIR/configFiles
tar xf ~/configFiles.tar -C $SLURM_TMPDIR/configFiles --strip-components=4
#strip components is so it doesn't create a ton of extra directories
#Do the above for the config files when training the abundance model
cd /home/tristanb/projects/def-pjmann/tristanb
#apptainer build psgImage.sif docker://nasapsg/psg-amd
python detectionModel.py


#/home/tristanb/projects/def-pjmann/tristanb/psgImage.sif
#scp -i C:\Users\Tristan\.ssh\sshCedar  tristanb@cedar.alliancecan.ca:/home/tristanb/projects/def-pjmann/tristanb/detectionModel.pt "C:\Users\Tristan\Downloads\HyPCAR3\"