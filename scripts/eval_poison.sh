#!/bin/bash
#SBATCH --job-name=use-wd
#SBATCH --time=1-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
# -- SBATCH --dependency=afterok:967617

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
# export MODEL_NAME='resnet18'
cd $SCRIPT_DIR
# python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 40 \
# --poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons/untargeted \
# --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm

# python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 40 \
# --poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted \
# --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm

# python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 40 \
# --poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted_adv_resnet18_iter=10 \
# --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm

# python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 40 \
# --poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted_adv_resnet18_iter=250 \
# --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm

# python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 40 \
# --poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons/untargeted_adv_resnet18_iter=250 \
# --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm

# python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 40 \
# --poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted_resnet18_iter=250 \
# --cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm

export POISON_DATASET_DIR='/vulcanscratch/psando/untrainable_datasets/adv_poisons/fresh_craft'
export MODEL_NAME='ResNet18'

python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 100 \
--cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm --workers 4 --lr 0.025 --use_wd

