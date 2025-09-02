#!/bin/bash

# Your Slurm setup: partition, number of nodes, etc.
# environment activation
# cd to the project root

# Set the path to save checkpoints
OUTPUT_DIR='/VideoMAE_logs/train_logs/my_pretrain/bl1_large_bdd-capdata_lightcrop_b100x8_mask075'
# path to data sets
DATA_PATH1='/scratch-nvme/ml-datasets/bdd100k/videos'
DATA_PATH2="/datasets/LOTVS-DADA/CAP-DATA"
# path to pretrained model
#
MODEL_PATH='/VideoMAE_logs/pretrained/VideoMAE/k400_vitl/checkpoint.pth'

# in case you use multiple nodes, make sure your setup is correct
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 hostname --ip-address)
export MASTER_PORT=29604  # Use a high, non-conflicting port
echo "RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK NTASKS=$SLURM_NTASKS"

# Normally, we use ViT-S based models with 1-GPU (H100) batch size 200.
# This option uses ViT-B model that requires 2x memory, so the base batch size is 100.
# So, to keep the original settings - total batch size 800, we will use 2 nodes instead of 1
srun python run_mae_double_pretraining.py \
    --data_set1 BDD100K \
    --data_path1 ${DATA_PATH1} \
    --sampling_rate1 16 \
    --data_set2 CAP-DATA \
    --data_path2 ${DATA_PATH2} \
    --sampling_rate2 1 \
    --mask_type tube \
    --mask_ratio 0.75 \
    --tubelet_size 2 \
    --model pretrain_videomae_large_patch16_224 \
    --from_ckpt ${MODEL_PATH} \
    --decoder_depth 12 \
    --batch_size1 60 \
    --batch_size2 40 \
    --num_frames 16 \
    --transforms_finetune_align \
    --nb_samples_per_epoch 1000000 \
    --num_workers 14 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 20 \
    --save_ckpt_freq 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lr 3e-4 \
    --min_lr 3e-5 \
