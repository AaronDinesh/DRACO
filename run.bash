srun --environment=$HOME/DRACO/GAN.toml -A sk030 -p normal -t 02:30:00 -G 1 --job-name=gan_uv \
  --output=./logs/gan_%j.out --error=./logs/gan_%j.err \
  bash -lc 'cd /users/adinesh/DRACO && source .venv/bin/activate && uv run main.py \
    --input-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy \
    --output-maps=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/Maps_Mstar_IllustrisTNG_SB28_z=0.00.npy \
    --cosmos-params=/capstor/store/cscs/ska/sk030/camels_multifield_dataset/params_SB28_IllustrisTNG.txt \
    --checkpoint-dir=/capstor/store/cscs/ska/sk030/gan_checkpoints/run_1'
