#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=advpoison-imagenet-v2
#SBATCH --time=3-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%j-%x.out
#SBATCH --array=0
#--SBATCH --output /dev/null
#--SBATCH --dependency=afterok:
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

set -x

python anneal.py --recipe targeted \
                 --eps 8 \
                 --budget 1.0 \
                 --dataset ImageNetMini \
                 --data_path /vulcanscratch/psando/imagenet/ \
                 --checkpoint_directory /fs/vulcan-projects/stereo-detection/unlearnable-ds-neurips-23/adversarial_poisons_checkpoints \
                 --save_final_checkpoint \
                 --poison_partition 25000 \
                 --save poison_dataset \
                 --poison_path /fs/vulcan-projects/stereo-detection/psando_poisons/paper/imagenet-100class/error-max-v2 \
                 --restarts 8 \
                 --resume resume_logs/ \
                 --resume_idx ${SLURM_ARRAY_TASK_ID} \
                 --attackoptim PGD
