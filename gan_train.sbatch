#!/bin/bash
#SBATCH --job-name=gan_test_adinesh
#SBATCH --time=00:30:00
#SBATCH --environment=GAN
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=logs/test_gan_%j.log
#SBATCH --error=logs/test_gan_%j.log
#SBATCH --partition=debug


cd ~/DRACO
uv run main.py --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mstar_IllustrisTNG_SB28_z=0.00.npy --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt --checkpoint-dir=/capstor/store/cscs/ska/sk030/gan_checkpoints/run_1
