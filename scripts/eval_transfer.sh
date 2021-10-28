#!/bin/bash
#SBATCH --job-name=eval-transfer
#SBATCH --time=12:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
#SBATCH --dependency=afterok:965252

set -x

export WORK_DIR="/scratch0/slurm_${SLURM_JOBID}"
export SCRIPT_DIR="/cfarhomes/psando/Documents/adversarial_poisons"

# Set environment 
mkdir $WORK_DIR
python3 -m venv ${WORK_DIR}/tmp-env
source ${WORK_DIR}/tmp-env/bin/activate
pip3 install --upgrade pip
pip3 install -r ${SCRIPT_DIR}/requirements.txt

# Train and Save
cd $SCRIPT_DIR

# python poison_evaluation/eval_transferability.py /vulcanscratch/psando/untrainable_datasets/adv_poisons/untargeted --disable_tqdm
# python poison_evaluation/eval_transferability.py /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted --disable_tqdm
# python poison_evaluation/eval_transferability.py /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted_adv_resnet18_iter=10 --disable_tqdm


# python poison_evaluation/eval_transferability.py /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted_adv_resnet18_iter=250 --disable_tqdm
# python poison_evaluation/eval_transferability.py /vulcanscratch/psando/untrainable_datasets/adv_poisons/untargeted_adv_resnet18_iter=250 --disable_tqdm

python poison_evaluation/eval_transferability.py /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted_resnet18_iter=250 --disable_tqdm