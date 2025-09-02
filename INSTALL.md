# VideoMAE Installation

The codebase is mainly built with following libraries:

- Python 3.6 or higher

- [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision). <br>

- [timm==0.4.12](https://github.com/rwightman/pytorch-image-models)
  `conda install -c conda-forge timm=0.4.12`

- [deepspeed==0.5.8](https://github.com/microsoft/DeepSpeed)

  `DS_BUILD_OPS=1 pip install deepspeed`

  **Comment**: In case the installation fails, remove the flag and simply do `pip install deepspeed`. In case of an error related to ds_kernels, install this first: 

  `pip install deepspeed-kernels` ([DeepSpeed-Kernels repo](https://github.com/microsoft/DeepSpeed-Kernels))

- [TensorboardX](https://github.com/lanpa/tensorboardX)

- [decord](https://github.com/dmlc/decord)

- [einops](https://github.com/arogozhnikov/einops)

- `pip install tensorboardX decord einops opencv-python scipy pandas`
- `pip install scikit-learn matplotlib seaborn torchmetrics natsort tqdm psutil`
- (optional) [FlashAttention](https://github.com/Dao-AILab/flash-attention)

### Note:
- You can run the model using only pytorch, use `run_inference_simple.py` as an example. However, you might need other libraries for data processing.
- We recommend you to use **`PyTorch >= 1.8.0`**. We used `2.5.1`. 
- We ran our code on NVIDIA A100 (40 GB) and NVIDIA H100 (94 GB) with CUDA 12.1.0 and 12.4.0. 

