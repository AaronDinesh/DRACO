#!/bin/bash
#SBATCH --job-name=gan_eval_noise
#SBATCH --time=04:00:00
#SBATCH --environment=/users/adinesh/DRACO/GAN.toml
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=/users/adinesh/DRACO/logs/eval_gan_noise_out_%j.log
#SBATCH --error=/users/adinesh/DRACO/logs/eval_gan_noise_err_%j.log
#SBATCH --partition=normal

RUN_NUMBER=$1
TYPE=$2
cd /users/adinesh/DRACO
nvidia-smi

python evaluate_gan.py \
  --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy \
  --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_${TYPE}_IllustrisTNG_SB28_z=0.00.npy \
  --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt \
  --generator-checkpoint-path=/capstor/store/cscs/ska/sk030/gan_checkpoints/${TYPE}/no_noise/run_${RUN_NUMBER}/Final_Generator \
  --discriminator-checkpoint-path=/capstor/store/cscs/ska/sk030/gan_checkpoints/${TYPE}/no_noise/run_${RUN_NUMBER}/Final_Discriminator \
  --generator-noise-checkpoint-path=/capstor/store/cscs/ska/sk030/gan_checkpoints/${TYPE}/with_noise/run_${RUN_NUMBER}/Final_Generator \
  --discriminator-noise-checkpoint-path=/capstor/store/cscs/ska/sk030/gan_checkpoints/${TYPE}/with_noise/run_${RUN_NUMBER}/Final_Discriminator \
  --output-dir=/capstor/store/cscs/ska/sk030/evaluation_runs/gan_noise/run_${RUN_NUMBER} \
  --field-name="${TYPE} Field (noise)" \
  --batch-size=64 \
  --img-channels=1 \
  --power-spectrum-bins=64 \
  --test-ratio=0.2 \
  --add-noise
