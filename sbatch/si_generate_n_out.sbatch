#!/bin/bash
#SBATCH --job-name=si_generate_adinesh
#SBATCH --time=24:00:00
#SBATCH --environment=/users/adinesh/DRACO/GAN.toml
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=/users/adinesh/DRACO/logs/si_generate_out_%j.log
#SBATCH --error=/users/adinesh/DRACO/logs/si_generate_err_%j.log
#SBATCH --partition=normal

RUN_NUMBER=$1
SAMPLE_IDX=${2:-0}
TYPE=$3

cd /users/adinesh/DRACO
# source .venv/bin/activate
export PYTHONPATH="/users/adinesh/DRACO:${PYTHONPATH}"
nvidia-smi

python generation/si_generate_n_out.py \
  --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy \
  --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt \
  --target-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_${TYPE}_IllustrisTNG_SB28_z=0.00.npy \
  --velocity-checkpoint-path=/capstor/store/cscs/ska/sk030/si_checkpoints/${TYPE}/run_${RUN_NUMBER}/velocity_final \
  --score-checkpoint-path=/capstor/store/cscs/ska/sk030/si_checkpoints/${TYPE}/run_${RUN_NUMBER}/score_final \
  --output-dir=/capstor/store/cscs/ska/sk030/evaluation_runs/si/${TYPE}/run_${RUN_NUMBER}/generated_samples/sample_${SAMPLE_IDX} \
  --sample-idx=${SAMPLE_IDX} \
  --split=test \
  --n-samples=1
