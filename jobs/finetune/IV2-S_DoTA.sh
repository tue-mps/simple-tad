#!/bin/bash

# Your Slurm setup: partition, number of nodes, etc.
# environment activation
# cd to the project root

# Set the path to save checkpoints
OUTPUT_DIR='/VideoMAE_logs/train_logs/finetune/DoTA/iv2-small_dota_lr1e3_b56x1_dsampl1val1_ld06_aam6n3'
# path to data set 
DATA_PATH='/datasets/DoTA'
# path to pretrain model
MODEL_PATH='/VideoMAE_logs/pretrained/InternVideo/pretr_s14_single_dist1B/IntVid2_s14_single_dist1B.bin'


# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32

# NOTE THAT THE PYTHON SCRIPT IS DIFFERENT, and the model name! Also, num_F=frames and view_fps. This is the only difference
torchrun --nproc_per_node=1 \
    --master_port 12459 \
    other_models/InternVideo2_single_modality/iv2_sm_run_frame_finetuning.py \
    --model internvideo2_small_patch14_224 \
    --no_cls_in_ckpt \
    --data_set DoTA \
    --loss crossentropy \
    --nb_classes 2 \
    --tubelet_size 1 \
    --no_use_decord \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 56 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 8 \
    --view_fps 5 \
    --sampling_rate 1 \
    --sampling_rate_val 1 \
    --nb_samples_per_epoch 50000 \
    --num_workers 8 \
    --opt adamw \
    --lr 1e-3 \
    --min_lr 1e-6 \
    --warmup_lr 1e-6 \
    --warmup_epochs 5 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --layer_decay 0.6 \
    --aa rand-m6-n3-mstd0.5-inc1 \
    --head_drop_path 0.1 \
    --epochs 50 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed \
    --zero_stage 0 \
    --seed 42
