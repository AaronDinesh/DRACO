#!/bin/bash
#SBATCH --job-name=si_eval
#SBATCH --time=24:00:00
#SBATCH --environment=/users/adinesh/DRACO/GAN.toml
#SBATCH --account=sk030
#SBATCH -G 1
#SBATCH --output=/users/adinesh/DRACO/logs/eval_si_trace_out_%j.log
#SBATCH --error=/users/adinesh/DRACO/logs/eval_si_trace_err_%j.log
#SBATCH --partition=normal

RUN_NUMBER=$1
TYPE=$2
SAMPLE=$3
cd /users/adinesh/DRACO
nvidia-smi

python generation/trace_si_sample.py \
  --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy \
  --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_${TYPE}_IllustrisTNG_SB28_z=0.00.npy \
  --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt \
  --velocity-checkpoint-path=/capstor/store/cscs/ska/sk030/si_checkpoints/${TYPE}/run_${RUN_NUMBER}/velocity_final \
  --score-checkpoint-path=/capstor/store/cscs/ska/sk030/si_checkpoints/${TYPE}/run_${RUN_NUMBER}/score_final \
  --output-dir=/capstor/store/cscs/ska/sk030/evaluation_runs/si/${TYPE}/run_${RUN_NUMBER} \
  --img-channels=1 \
  --integrator-steps=3000 \
  --n-save=4 \
  --n-likelihood=1 \
  --test-ratio=0.2 \
  --sample-idx=$3 \
  --trace-dir=/capstor/store/cscs/ska/sk030/evaluation_runs/si/${TYPE}/run_${RUN_NUMBER}/traced_gen
