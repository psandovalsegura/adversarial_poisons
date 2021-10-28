#!/bin/bash
#SBATCH --job-name=anneal
#SBATCH --time=3-00:00:00
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
export CKPT_DIR="/vulcanscratch/psando/cifar_model_ckpts"

# Set environment 
mkdir $WORK_DIR
python3 -m venv ${WORK_DIR}/tmp-env
source ${WORK_DIR}/tmp-env/bin/activate
pip3 install --upgrade pip
pip3 install -r ${SCRIPT_DIR}/requirements.txt


# python anneal.py --net ResNet18 --dataset CIFAR10 --data_path /vulcanscratch/psando/cifar-10/ \
# --recipe targeted --eps 8 --budget 1.0 --save poison_dataset \
# --poison_path /vulcanscratch/psando/untrainable_datasets/adv_poisons/targeted \
# --attackoptim PGD --attackiter 10

export POISON_DATASET_DIR='/vulcanscratch/psando/untrainable_datasets/adv_poisons/fresh_craft'
export MODEL_NAME='ResNet18'
export RECIPE='targeted'
export ATTACKITER='250'

# adversarial model surrogate
# python anneal.py --net $MODEL_NAME --dataset CIFAR10 --data_path /vulcanscratch/psando/cifar-10/ \
# --recipe $RECIPE --eps 8 --budget 1.0 --save poison_dataset \
# --cifar_ckpt_dir /vulcanscratch/psando/cifar_model_ckpts/ --cifar_adv_ckpt_dir /vulcanscratch/psando/cifar_model_ckpts/adv \
# --adv_pretrained --poison_path ${POISON_DATASET_DIR}/${RECIPE}_adv_${MODEL_NAME}_iter=${ATTACKITER} \
# --attackoptim PGD --attackiter $ATTACKITER

# standard model surrogate (pretrained model trained for 200 epochs)
# python anneal.py --net $MODEL_NAME --dataset CIFAR10 --data_path /vulcanscratch/psando/cifar-10/ \
# --recipe $RECIPE --eps 8 --budget 1.0 --save poison_dataset \
# --cifar_ckpt_dir /vulcanscratch/psando/cifar_model_ckpts/ --cifar_adv_ckpt_dir /vulcanscratch/psando/cifar_model_ckpts/adv \
# --pretrained --poison_path ${POISON_DATASET_DIR}/${RECIPE}_${MODEL_NAME}_iter=${ATTACKITER} \
# --attackoptim PGD --attackiter $ATTACKITER

python anneal.py --net $MODEL_NAME --dataset CIFAR10 --data_path /vulcanscratch/psando/cifar-10/ \
--recipe $RECIPE --eps 8 --budget 1.0 --save poison_dataset \
--cifar_ckpt_dir /vulcanscratch/psando/cifar_model_ckpts/ --cifar_adv_ckpt_dir /vulcanscratch/psando/cifar_model_ckpts/adv \
--poison_path ${POISON_DATASET_DIR}/${RECIPE}_${MODEL_NAME}_iter=${ATTACKITER} \
--attackoptim PGD --attackiter $ATTACKITER