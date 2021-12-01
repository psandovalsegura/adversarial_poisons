#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=train-settings-targeted
#SBATCH --time=3-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
# -- SBATCH --dependency=afterok:

set -x

export WORK_DIR="/scratch0/slurm_${SLURM_JOBID}"
export SCRIPT_DIR="/cfarhomes/psando/Documents/adversarial_poisons"
export CKPT_DIR="/vulcanscratch/psando/cifar_model_ckpts"

# Set environment 
mkdir $WORK_DIR
python3 -m venv ${WORK_DIR}/tmp-env
source ${WORK_DIR}/tmp-env/bin/activate
pip3 install --upgrade pip
pip3 install -r ${SCRIPT_DIR}/requirements.txt

export POISON_DATASET_DIR='/vulcanscratch/psando/untrainable_datasets/adv_poisons/fresh_craft'
export MODEL_NAME='ResNet18'
export POISON_NAME='targeted_ResNet18_iter=250'

# Evaluate LR schedule and weight decay training settings

# WD: False, LR Scheduler: False
# python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 100 \
# --poison_path ${POISON_DATASET_DIR}/${POISON_NAME} \
# --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm --workers 4 --lr 0.025 

# WD: False, LR Scheduler: True
# python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 100 \
# --poison_path ${POISON_DATASET_DIR}/${POISON_NAME} \
# --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm --workers 4 --lr 0.025 --use_scheduler

# WD: True, LR Scheduler: False
python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 100 \
--poison_path ${POISON_DATASET_DIR}/${POISON_NAME} \
--cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm --workers 4 --lr 0.025 --use_wd

# WD: True, LR Scheduler: True
python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 100 \
--poison_path ${POISON_DATASET_DIR}/${POISON_NAME} \
--cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm --workers 4 --lr 0.025 --use_scheduler --use_wd