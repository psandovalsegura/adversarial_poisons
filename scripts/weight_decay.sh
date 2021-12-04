#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=weight-4
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

export POISON_DATASET_DIR='/vulcanscratch/psando/untrainable_datasets/adv_poisons/fresh_craft'
export MODEL_NAME='ResNet18'
export EPS='4'

# Set environment 
mkdir $WORK_DIR
python3 -m venv ${WORK_DIR}/tmp-env
source ${WORK_DIR}/tmp-env/bin/activate
pip3 install --upgrade pip
pip3 install -r ${SCRIPT_DIR}/requirements.txt

# Use class-wise poisons
# 8/255: classwise_random_eps=8
# 64/255: classwise_random_eps=64
declare -a WD_FLOATS=('0' '1e-4' '2e-4' '3e-4' '4e-4' '5e-4' '6e-4')
for WD_FLOAT in ${WD_FLOATS[@]}; do

# Evaluate poison
python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 100 \
--poison_path ${POISON_DATASET_DIR}/classwise_random_eps=8 \
--cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm --workers 4 --lr 0.025 --use_scheduler --weight_decay ${WD_FLOAT}

done
