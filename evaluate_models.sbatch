#!/bin/bash
#SBATCH --job-name=model_eval_adinesh
#SBATCH --time=04:00:00
#SBATCH --environment=/users/adinesh/DRACO/GAN.toml
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=/users/adinesh/DRACO/logs/eval_models_out_%j.log
#SBATCH --error=/users/adinesh/DRACO/logs/eval_models_err_%j.log
#SBATCH --partition=normal

RUN_NUMBER=$1

cd /users/adinesh/DRACO
nvidia-smi

python evaluate_models.py \
  --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy \
  --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_B_IllustrisTNG_SB28_z=0.00.npy \
  --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt \
  --generator-checkpoint-path=/capstor/store/cscs/ska/sk030/gan_checkpoints/B/no_noise/run_${RUN_NUMBER}/Final_Generator \
  --discriminator-checkpoint-path=/capstor/store/cscs/ska/sk030/gan_checkpoints/B/no_noise/run_${RUN_NUMBER}/Final_Discriminator \
  --velocity-checkpoint-path=/capstor/store/cscs/ska/sk030/si_checkpoints/Mtot/run_${RUN_NUMBER}/velocity_final \
  --score-checkpoint-path=/capstor/store/cscs/ska/sk030/si_checkpoints/Mtot/run_${RUN_NUMBER}/score_final \
  --output-dir=/capstor/store/cscs/ska/sk030/evaluation_runs/run_${RUN_NUMBER} \
  --field-name="B Field" \
  --batch-size=64 \
  --img-channels=1 \
  --power-spectrum-bins=64 \
  --integrator-steps=500 \
  --n-save=4 \
  --n-likelihood=1
