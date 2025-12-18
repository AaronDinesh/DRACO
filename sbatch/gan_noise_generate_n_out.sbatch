#!/bin/bash
#SBATCH --job-name=gan_generate_adinesh
#SBATCH --time=24:00:00
#SBATCH --environment=/users/adinesh/DRACO/GAN.toml
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=/users/adinesh/DRACO/logs/gan_generate_out_%j.log
#SBATCH --error=/users/adinesh/DRACO/logs/gan_generate_err_%j.log
#SBATCH --partition=normal

RUN_NUMBER=$1
SAMPLE_IDX=${2:-0}
TYPE=$3

cd /users/adinesh/DRACO
export PYTHONPATH="/users/adinesh/DRACO:${PYTHONPATH}"
nvidia-smi



python generation/gan_generate_n_out.py \
  --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy \
  --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt \
  --target-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_${TYPE}_IllustrisTNG_SB28_z=0.00.npy \
  --generator-checkpoint-path=/capstor/store/cscs/ska/sk030/gan_checkpoints/${TYPE}/with_noise/run_${RUN_NUMBER}/Final_Generator \
  --output-dir=/capstor/store/cscs/ska/sk030/evaluation_runs/gan/${TYPE}/with_noise/run_${RUN_NUMBER}/generated_samples/sample_${SAMPLE_IDX} \
  --sample-idx=${SAMPLE_IDX} \
  --n-samples=15 \
  --img-channels=1 \
  --transform-name=log10 \
  --add-noise \
  --split=test \
  --n-samples=1
