#!/bin/bash
#SBATCH --job-name=si_test_adinesh
#SBATCH --time=24:00:00
#SBATCH --environment=/users/adinesh/DRACO/GAN.toml
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=/users/adinesh/DRACO/logs/test_si_out_%j.log
#SBATCH --error=/users/adinesh/DRACO/logs/test_si_err_%j.log
#SBATCH --partition=normal

RUN_NUMBER=$1
TYPE=$2

cd /users/adinesh/DRACO
# source .venv/bin/activate
export PYTHONPATH="/users/adinesh/DRACO:${PYTHONPATH}"
nvidia-smi


python training/SI_train.py --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy \
                   --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_${TYPE}_IllustrisTNG_SB28_z=0.00.npy \
                   --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt \
                   --checkpoint-dir=/capstor/store/cscs/ska/sk030/si_checkpoints/${TYPE}/run_${RUN_NUMBER} \
                   --wandb-run-name=nnx-si-256-run-${TYPE}-${RUN_NUMBER} \
                   --batch-size=32 \
                   --epochs=200

# python training/SI_train.py --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy \
#                    --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_${TYPE}_IllustrisTNG_SB28_z=0.00.npy \
#                    --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt \
#                    --checkpoint-dir=/capstor/store/cscs/ska/sk030/si_checkpoints/${TYPE}/run_${RUN_NUMBER} \
#                    --wandb-run-name=nnx-si-256-run-${TYPE}-${RUN_NUMBER} \
#                    --batch-size=32 \
#                    --epochs=200 \
#                    --velocity-checkpoint-path=/capstor/store/cscs/ska/sk030/si_checkpoints/${TYPE}/run_${RUN_NUMBER}/velocity_final \
#                    --score-checkpoint-path=/capstor/store/cscs/ska/sk030/si_checkpoints/${TYPE}/run_${RUN_NUMBER}/score_final
