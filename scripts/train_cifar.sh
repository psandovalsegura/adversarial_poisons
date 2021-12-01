#!/bin/bash
#SBATCH --job-name=train-cifar-model
#SBATCH --time=1-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
# -- SBATCH --dependency=afterok:

set -x

export WORK_DIR="/scratch0/slurm_${SLURM_JOBID}"
export SCRIPT_DIR="/cfarhomes/psando/Documents/adversarial_poisons"
export CKPT_DIR="/vulcanscratch/psando/cifar_model_ckpts"

# Set environment for attack
mkdir $WORK_DIR
python3 -m venv ${WORK_DIR}/tmp-env
source ${WORK_DIR}/tmp-env/bin/activate
pip3 install --upgrade pip
pip3 install -r ${SCRIPT_DIR}/requirements.txt

# Train and Save
export MODEL_NAME='ResNet18'
cd $SCRIPT_DIR
python poison_evaluation/main.py --model_name $MODEL_NAME \
--runs 1 --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm