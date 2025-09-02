#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --time=60:00:00
#SBATCH --output=jobs_outs_extra/umt_dada_%j.out

# For H100 nodes:
export NCCL_SOCKET_IFNAME="eno1np0"
#export NCCL_DEBUG=INFO
#export CUDA_LAUNCH_BLOCKING=1

module load 2023
module load Anaconda3/2023.07-2

export OMP_NUM_THREADS=18
export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.4.0/

__conda_setup="$('/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /home/eorozco/miniconda3/envs/vid

cd /home/eorozco/sveta/repos/VideoMAE

# Set the path to save checkpoints
OUTPUT_DIR='/projects/0/prjs1424/sveta/VideoMAE_logs/train_logs/finetune/extra_models/umt/win16fps10_base_dada-ref_lr5e4_b56x1_dsampl1val1_ld06_aam6n3'
# path to data set 
DATA_PATH='/projects/0/prjs1424/sveta/RiskNetData/LOTVS-DADA/DADA2K/'
# path to pretrain model
# '/projects/0/prjs1424/sveta/VideoMAE_logs/pretrained/VideoMAE/k400_vitl/checkpoint.pth'  # 1 LARGE
# '/projects/0/prjs1424/sveta/VideoMAE_logs/pretrained/VideoMAE/k400_vitb/checkpoint_ep1600pt.pth'  # 1 BASE
# '/projects/0/prjs1424/sveta/VideoMAE_logs/pretrained/VideoMAE2_distill/vit_b_k710_dl_from_giant.pth'  # 2 BASE
MODEL_PATH='/projects/0/prjs1424/sveta/VideoMAE_logs/pretrained/UMT/b16_ptk710_f8_res224.pth'


# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32

# 2 hrs per epoch
torchrun --nproc_per_node=1 \
    --master_port 12622 \
    models/UMT/run_frame_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set DADA2K \
    --loss crossentropy \
    --nb_classes 2 \
    --tubelet_size 1 \
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
