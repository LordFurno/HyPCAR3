module load python/3.12
module load mpi4py/3.1.6

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r /home/tristanb/projects/def-pjmann/tristanb/dataGenerationRequirements.txt
pip install torch

pip install requests
cd /home/tristanb/projects/def-pjmann/tristanb
export PSM2_CUDA=0
#python dataGenerationFix.py
python psgCallTest.py
/home/tristanb/projects/def-pjmann/tristanb/workingDirectory/working-i.txt