#!/bin/bash

# Your Slurm setup: partition, number of nodes, etc.
# environment activation
# cd to the project root

# Set the path to save checkpoints
OUTPUT_DIR='logs/my_pretrain/bdd100k_extra_pretrain_vits/lightcrop_from-k400_full_regular_b200x4_mask075'
# path to data set 
DATA_PATH='/scratch-nvme/ml-datasets/bdd100k/videos'
# path to pretrain model
MODEL_PATH='logs/pretrained/k400_vits/videomae_vits_k400_pretrain_ckpt.pth'

# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32
# srun python run_frame_finetuning.py \
torchrun --nproc_per_node=4 \
    --master_port 12474 \
    run_mae_pretraining.py \
    --data_set BDD100K \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio 0.75 \
    --model pretrain_videomae_small_patch16_224 \
    --from_ckpt ${MODEL_PATH} \
    --decoder_depth 4 \
    --batch_size 200 \
    --num_frames 16 \
    --sampling_rate 16 \
    --transforms_finetune_align \
    --nb_samples_per_epoch 1000000 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 20 \
    --save_ckpt_freq 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lr 3e-4 \
    --min_lr 3e-5 \
