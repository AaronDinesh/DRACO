#!/bin/bash
#SBATCH --job-name=gan_test_adinesh
#SBATCH --time=00:10:00
#SBATCH --environment=$HOME/DRACO/GAN.toml
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=$HOME/DRACO/logs/test_gan_%j.log
#SBATCH --error=$HOME/DRACO/logs/test_gan_%j.log
#SBATCH --partition=debug



cd /users/adinesh/DRACO
source .venv/bin/activate
uv run main.py --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mtot_IllustrisTNG_SB28_z=0.00.npy --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt --checkpoint-dir=/capstor/store/cscs/ska/sk030/gan_checkpoints/run_2
