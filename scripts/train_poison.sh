#!/bin/bash
#SBATCH --job-name=ckpt-poison
#SBATCH --time=1-12:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
#SBATCH --dependency=afterok:966871

set -x

export WORK_DIR="/scratch0/slurm_${SLURM_JOBID}"
export SCRIPT_DIR="/cfarhomes/psando/Documents/adversarial_poisons"
export CKPT_DIR="/vulcanscratch/psando/cifar_model_ckpts/poisoned/targeted_250"

# Set environment for attack
mkdir $WORK_DIR
python3 -m venv ${WORK_DIR}/tmp-env
source ${WORK_DIR}/tmp-env/bin/activate
pip3 install --upgrade pip
pip3 install -r ${SCRIPT_DIR}/requirements.txt

# Train and Save
export MODEL_NAME='resnet18'
cd $SCRIPT_DIR

python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 100 --runs 1 --workers 2 \
--poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons/fresh_craft/targeted_ResNet18_iter=250 \
--ckpt_dir ${CKPT_DIR} \
--cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm

