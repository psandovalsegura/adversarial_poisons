#!/bin/bash
#SBATCH --job-name=adv-train-cifar-model
#SBATCH --time=1-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
# -- SBATCH --dependency=afterok:

set -x

export WORK_DIR="/scratch0/slurm_${SLURM_JOBID}"
export SCRIPT_DIR="/cfarhomes/psando/Documents/adversarial_poisons"
export CKPT_DIR="/vulcanscratch/psando/cifar_model_ckpts/adv"

# Set environment for attack
mkdir $WORK_DIR
python3 -m venv ${WORK_DIR}/tmp-env
source ${WORK_DIR}/tmp-env/bin/activate
pip3 install --upgrade pip
pip3 install -r ${SCRIPT_DIR}/requirements.txt

# Train and Save
export MODEL_NAME='densenet121'
cd $SCRIPT_DIR
python poison_evaluation/adv_train.py --model_name $MODEL_NAME --ckpt_dir $CKPT_DIR \
--cifar_path /vulcanscratch/psando/cifar-10 
