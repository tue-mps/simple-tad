# Domain-Adaptive Pre-training (DAPT) and Fine-tuning

You will find Slurm scripts for DAPT and fine-tuning in the `jobs` folder, with the settings that we used. 
To run the code locally, just use torchrun command with the corresponding arguments.

In the example scripts, you will find links to the pretrained weights of the models we used.

To use ViViT, first reformat their checkpoint so it matches our model structure: `other_models/ViViT/ckpt_vivit.py`

Some options:

- if your training was interrupted after some number of epochs, the option `auto_resume` (True by default) will continue the training using `checkpoint-last` folder.
- `sampling_rate` is the rate of subsampling performed over all the possible samples (input windows). For very large datasets, or datasets with FPS more than 10, it may be helpful to have sampling rate 2 or 3 for the training phase (if no `nb_samples_per_epoch` set), validation, or both. During evaluation, it should be set to 1.
- `nb_samples_per_epoch` is how many samples (input windows) you want to have during training. If not set, the whole dataset will be used.
- `num_frames` (default 16) and `view_fps` (default 10) define the input window: the number of frames sampled, and with what FPS. So, 16 frames sampled with 10 FPS, and 8 frames sampled with 5 FPS cover the same time range.

## DAPT

Scripts are in `jobs/dapt`. We set cosine learning rate scheduler for 20 epochs and stop the training after 12 epochs (or 23 for Kinetics-700). 

Please note that for BDD and BDD+CAP-DATA pretraining, we use: 
  - 1M samples per epoch    -> 1250 steps per epoch (over all GPUs in total) -> 20 epochs to achieve 25K steps

Training on Kinetics, we have only ~0.5M videos, and we sample one video once per epoch by default, so:
  - 536685 samples per epoch -> 670 steps per epoch (over all GPUs in total) -> 38 ep to achieve 25K steps
Then we stopped training after 12 epochs, and here it will be 22-23 epochs

Checkpoints we used to initialize models for DAPT are from [VideoMAE-ModelZOO-Kinetics400](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#kinetics-400):
- VideoMAE-S 1600 epochs
- VideoMAE-B 1600 epochs
- VideoMAE-L 1600 epochs

## Fine-tuning

Scripts are in `jobs/finetune`.
We used different learning rate for different model sizes: 1e-3 for Small, and 5e-4 for all other variants. We used `num_frames=8`, and `view_fps=5` for UMT and InternVideo2, for all other models: `num_frames=16`, and `view_fps=10`. All other settings are the same. 

Note that for VideoMAE, VideoMAE2, ViViT, SMILE, UMT, SIGMA, MME, MGMAE, you can use the same script and only change the pre-trained weights that you initialize the model with. 
**For InternVideo2, MVD, and UMT, use their respective scripts and model names (check out the scripts provided).**

Checkpoints we used to initialize models for fine-tuning:

### VideoMAE 
Kinetics-400 [link](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#kinetics-400):
- VideoMAE-S 1600 epochs
- VideoMAE-B 800 epochs
- VideoMAE-B 1600 epochs
- VideoMAE-L 1600 epochs

### VideoMAE V2 
Distilled [link](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md#distillation):
- VideoMAEv2-S (vit_s_k710_dl_from_giant.pth)
- VideoMAEv2-B (vit_b_k710_dl_from_giant.pth)

### ViViT
See `other_models/ViViT/ckpt_vivit.py`

### MVD

Kinetics-400 [link](https://github.com/ruiwang2021/mvd/blob/main/MODEL_ZOO.md#kinetics-400)

We used all Small, Base and Large model variants.

### UMT

The Base checkpoint: [Pretrained models](https://github.com/OpenGVLab/unmasked_teacher/blob/main/single_modality/MODEL_ZOO.md#pretraining)

### InternVideo2

Distilled [link](https://github.com/OpenGVLab/unmasked_teacher/blob/main/single_modality/MODEL_ZOO.md#pretraining)

### SMILE

The Base checkpoint: [Kinetics-400 ](https://github.com/fmthoker/SMILE?tab=readme-ov-file#-kinetics-400)

### MME

The Base model, 1600 epochs, Kinetics-400: [link](https://github.com/XinyuSun/MME?tab=readme-ov-file#pre-trained-weight)

### SIGMA

The Base model, 1600 epochs, Kinetics-400, [Pre-trained](https://github.com/QUVA-Lab/SIGMA/blob/gh-pages/MODEL_ZOO.md#kinetics-400)

### MGMAE

The Base model, 1600 epochs, Kinetics-400: [Pre-trained models](https://github.com/MCG-NJU/MGMAE/blob/main/docs/MODEL_ZOO.md#pre-train)


