#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=advpoison-svhn
#SBATCH --time=1-12:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=slurm-%j-%x.out
#--SBATCH --array=1-5
#--SBATCH --output /dev/null
#--SBATCH --dependency=afterok:
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

set -x

python anneal.py --recipe targeted \
                 --eps 8 \
                 --budget 1.0 \
                 --dataset SVHN \
                 --data_path /vulcanscratch/psando/SVHN \
                 --checkpoint_directory /fs/vulcan-projects/stereo-detection/unlearnable-ds-neurips-23/adversarial_poisons_checkpoints \
                 --save_final_checkpoint \
                 --save poison_dataset \
                 --poison_path /fs/vulcan-projects/stereo-detection/psando_poisons/paper/svhn/linf/error-max \
                 --restarts 3 \
                 --attackoptim PGD
