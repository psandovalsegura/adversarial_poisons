#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=anneal-functional-C
#SBATCH --time=1-12:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
# -- SBATCH --dependency=afterok:

set -x
echo "anneal-functional-C"

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
export RECIPE='functional'
export ATTACKITER='100'
export ATTACKNAME='recoloradv'

# Craft poison
# python anneal.py --net $MODEL_NAME --dataset CIFAR10 --data_path /vulcanscratch/psando/cifar-10/ \
# --recipe $RECIPE --eps 8 --budget 1.0 --save poison_dataset \
# --cifar_ckpt_dir /vulcanscratch/psando/cifar_model_ckpts/ --cifar_adv_ckpt_dir /vulcanscratch/psando/cifar_model_ckpts/adv \
# --poison_path ${POISON_DATASET_DIR}/${RECIPE}_${MODEL_NAME}_name=${ATTACKNAME}_iter=${ATTACKITER} \
# --attackname ${ATTACKNAME} --attackiter $ATTACKITER --init zero 

# Evaluate poison
python poison_evaluation/main.py --model_name $MODEL_NAME --epochs 100 \
--poison_path ${POISON_DATASET_DIR}/${RECIPE}_${MODEL_NAME}_name=${ATTACKNAME}_iter=${ATTACKITER} \
--cifar_path /vulcanscratch/psando/cifar-10 --disable_tqdm --workers 4 --lr 0.025 --use_wd --use_scheduler

# Evaluate transferability
python poison_evaluation/eval_transferability.py ${POISON_DATASET_DIR}/${RECIPE}_${MODEL_NAME}_name=${ATTACKNAME}_iter=${ATTACKITER} --disable_tqdm --workers 4 