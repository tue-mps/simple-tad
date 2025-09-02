#!/bin/bash

# Your Slurm setup: partition, number of nodes, etc.
# environment activation
# cd to the project root

# Set the path to save checkpoints
OUTPUT_DIR='logs/my_pretrain/bl1_vits_k700/'
# path to Kinetics set (train.csv/val.csv/test.csv
DATA_PATH='/scratch-nvme/ml-datasets/kinetics/k700-2020'
# path to pretrain model
MODEL_PATH='logs/pretrained/k400_vits/videomae_vits_k400_pretrain_ckpt.pth'

# For BDD and BDD+CAP-DATA pretraining, we use: 
#   - 1M samples per epoch    -> 1250 steps per epoch (over all GPUs in total) -> 20 epochs to achieve 25K steps
# With Kinetics, we have only ~0.5M videos and we sample one video once per epoch be default, so:
#   - 536685 samples per epoch -> 670 steps per epoch (over all GPUs in total) -> 38 ep to achieve 25K steps
# Then we stopped training after 12 epochs, and here it will be 22-23 epochs
torchrun --nproc_per_node=4 \
    run_mae_pretraining.py \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio 0.9 \
    --model pretrain_videomae_small_patch16_224 \
    --from_ckpt ${MODEL_PATH} \
    --decoder_depth 4 \
    --batch_size 200 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 38 \
    --save_ckpt_freq 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lr 3e-4 \
    --min_lr 3e-5 \
