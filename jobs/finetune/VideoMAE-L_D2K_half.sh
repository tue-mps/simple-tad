#!/bin/bash

# Your Slurm setup: partition, number of nodes, etc.
# environment activation
# cd to the project root

# Set the path to save checkpoints
OUTPUT_DIR='/VideoMAE_logs/train_logs/finetune/DADA2K/vm1-large_dadah_lr5e4_b56x1_dsampl1val1_ld06_aam6n3'
# path to data set 
DATA_PATH='/datasets/LOTVS-DADA/DADA2K/'
# path to pretrained model
MODEL_PATH='/projects/0/prjs1424/sveta/VideoMAE_logs/pretrained/VideoMAE2_distill/vit_b_k710_dl_from_giant.pth'



# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32

# 2 hrs per epoch
torchrun --nproc_per_node=1 \
    --master_port 12478 \
    run_frame_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set DADA2K_half \
    --loss crossentropy \
    --nb_classes 2 \
    --tubelet_size 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 28 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --sampling_rate 1 \
    --sampling_rate_val 3 \
    --nb_samples_per_epoch 50000 \
    --num_workers 8 \
    --opt adamw \
    --lr 5e-4 \
    --min_lr 1e-6 \
    --warmup_lr 1e-6 \
    --warmup_epochs 5 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --layer_decay 0.6 \
    --aa rand-m6-n3-mstd0.5-inc1 \
    --epochs 50 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed \
    --seed 42
