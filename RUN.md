# Evaluation and Inference 

## Evaluation

### Evaluation over a dataset

To evaluate a model, simply use `eval` option. Make sure that you set the checkpoint path in `MODEL_PATH` and the directory where you want the results to be saved in `OUTPUT_DIR`

```bash
OUTPUT_DIR='/your_training_log_directory/results_15'
DATA_PATH='/datasets/DoTA'
MODEL_PATH='/your_training_log_directory/checkpoint-15.pth'

torchrun --nproc_per_node=1 \
    run_frame_finetuning.py \
    --eval \ 
    ...
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    ...
```

Note that you can use much larger batch size!

This will produce 2 files:
- `predictions.csv` with all the predictions that you can use later to calculate more metrics.
- `stats.txt` with the metrics we reported in the paper, and more.

Then, you can use your `OUTPUT_DIR` with results in scripts 
`anaysis/metrics_by_categories.py`,
`anaysis/metrics_dada.py`, `anaysis/metrics_dota.py`, `anaysis/vis_video_paper.py`

### Efficiency evaluation

Use `test_efficiency.py`. 

## Inference

We provide a script with a minimal example of how to prepare the frames and to run the model over the video: 
`run_inference.py`

Also, the standalone script: `run_inference_simple.py`

