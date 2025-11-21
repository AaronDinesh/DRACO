#!/bin/bash
#SBATCH --job-name=gan_test_adinesh
#SBATCH --time=24:00:00
#SBATCH --environment=/users/adinesh/DRACO/GAN.toml
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=/users/adinesh/DRACO/logs/test_gan_out_%j.log
#SBATCH --error=/users/adinesh/DRACO/logs/test_gan_err_%j.log
#SBATCH --partition=normal

RUN_NUMBER=$1


cd /users/adinesh/DRACO
# source .venv/bin/activate
nvidia-smi

python GAN_train.py --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_B_IllustrisTNG_SB28_z=0.00.npy --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt --checkpoint-dir=/capstor/store/cscs/ska/sk030/gan_checkpoints/B/no_noise/run_${RUN_NUMBER} --wandb-run-name=nnx-cgan-256-run-B-${RUN_NUMBER} --epochs=200
