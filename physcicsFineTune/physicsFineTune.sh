#!/bin/bash
#SBATCH --account=def-pjmann
#SBATCH --job-name=physicsFineTune
#SBATCH --output=physicsFineTune-%J.out
#SBATCH --time=12-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=sirfurno@gmail.com
#SBATCH --mail-type=ALL


module mpi4py/3.1.6
module load python/3.12 cuda cudnn
module load apptainer

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r /home/tristanb/projects/def-pjmann/tristanb/pureCNNRequirements.txt


mkdir $SLURM_TMPDIR/data
tar xf ~/data.tar -C $SLURM_TMPDIR/data --strip-components=4
#strip components is so it doesn't create a ton of extra directories
#Do the above for the config files when training the abundance model
mkdir $SLURM_TMPDIR/configFiles
tar xf ~/configFiles.tar -C $SLURM_TMPDIR/configFiles --strip-components=4

mkdir $SLURM_TMPDIR/workingDirectory
tar xf /home/tristanb/projects/def-pjmann/tristanb/workingDirectory.tar -C $SLURM_TMPDIR/workingDirectory --strip-components=1

cd /home/tristanb/projects/def-pjmann/tristanb

apptainer instance start \
  -B ~/my-var-run:/var/run:rw \
  -B ~/projects/def-pjmann/tristanb/tempConf.conf:/etc/httpd/conf/httpd.conf:rw \
  -B ~/projects/def-pjmann/tristanb/tempwww.conf:/etc/php-fpm.d/www.conf:rw \
  -B ~/apache_logs:/etc/httpd/logs:rw \
  -B ~/phpfpm_logs:/var/log/php-fpm:rw \
  -B ~/projects/def-pjmann/tristanb/tempResults:/var/www/html/results:rw \
  workingPSG.sif psg

apptainer exec instance://psg /usr/sbin/httpd -D FOREGROUND &
apptainer exec instance://psg /usr/sbin/php-fpm

python hypcarAbundanceModel.py

