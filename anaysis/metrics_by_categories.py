import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from anaysis.metrics import calculate_metrics, calculate_fn_group, calculate_fn_group_thresholds, calculate_MORE_metrics, THRESHOLDS
from anaysis.make_plots import train_logs


def save_metrics(anno_csv, preds_dir):
    pred_csv = os.path.join(preds_dir, "predictions.csv")
    out_results_group = os.path.join(preds_dir, "group_stats.csv")
    out_metrics_thresholded = os.path.join(preds_dir, "thresh_stats.csv")

    # 1 combine data
    df_preds = pd.read_csv(pred_csv)
    df_anno = pd.read_csv(anno_csv)

    # Prepare probs
    logits = df_preds[["logits_safe", "logits_risk"]].to_numpy()
    probs = np.exp(logits)/np.expand_dims(np.sum(np.exp(logits), axis=-1), axis=-1)
    probs = probs[:, 1]
    df_preds["probs"] = probs
    df_preds = df_preds.drop('logits_safe', axis=1)
    df_preds = df_preds.drop('logits_risk', axis=1)

    if "dota" in os.path.basename(preds_dir).lower():
        df_preds['filename'] = df_preds['filename'].apply(lambda x: f"{str(x).zfill(6)}.jpg" if isinstance(x, int) else x)
        ok_percent = 0.3
    elif "dada2k" in os.path.basename(preds_dir).lower():
        df_preds['filename'] = df_preds['filename'].apply(lambda x: f"{str(x).zfill(4)}.png" if isinstance(x, int) else x)
        if "movad" in preds_dir.lower():  # because 30  -> 10 FPS and frame-by-frame prediction
            ok_percent = 0.3
        elif "comparison_cc" in preds_dir.lower():  # because 30  -> 10 FPS
            ok_percent = 0.15
        else:
            ok_percent = 0.15
    else:
        raise ValueError(f"Unknown dataset in {os.path.basename(preds_dir).lower()}")

    annot_subset = df_anno[['clip', 'filename', 'ego', 'night', 'cat']]
    df_preds = df_preds.merge(annot_subset, on=['clip', 'filename'], how='left')

    if df_preds.isnull().values.any():
        num_rows = df_preds.shape[0]
        num_rows_with_nan = df_preds.isnull().any(axis=1).sum()
        percent = num_rows_with_nan / num_rows
        print(f"There are {percent*100:.2f}% missing values in the DataFrame while normal max percent is {ok_percent}. {preds_dir}")
        if percent < ok_percent:
            print("It's okay. Remove invalid rows and proceed...")
            df_preds = df_preds.dropna()
        else:
            print("It's not okay. Halt.")
            exit(0)
    else:
        print("No missing values found!")

    del annot_subset
    del df_anno

    # Data is ready
    results = []
    thresh_results = []
    empty_list = [-1 for _ in THRESHOLDS]

    # First, get general (non-grouped) metrics
    thresh_metrics = calculate_MORE_metrics(preds=df_preds["probs"], labels=df_preds["label"])
    mcc_thresholded_vals, p_thresholded_vals, r_thresholded_vals, acc_thresholded_vals, f1_thresholded_vals = thresh_metrics[-5:]
    thresh_results.append(
        ["all_samples"] + mcc_thresholded_vals + p_thresholded_vals + r_thresholded_vals + 
        acc_thresholded_vals + f1_thresholded_vals
        )

    # 2 get metrics
    df_preds["cat"] = df_preds["cat"].astype(int)
    df_preds["ego"] = df_preds["ego"].astype(int)
    df_preds["night"] = df_preds["night"].astype(int)

    categories = natsorted(df_preds["cat"].unique())
    for cat in categories:
        df_group = df_preds[df_preds["cat"] == cat]
        res = calculate_fn_group_thresholds(probs=df_group["probs"], labels=df_group["label"])
        res = np.array(res)
        res = res if cat == 0 else 1.-res
        thresh_results.append([f"cat_{cat}"] + empty_list + empty_list + list(res) + empty_list + empty_list)

    df_group = df_preds[df_preds['ego'] == 1]
    auroc = roc_auc_score(df_group["label"], df_group["probs"])
    print(f"auroc ego: {auroc}")
    res_ = calculate_MORE_metrics(preds=df_group["probs"], labels=df_group["label"])
    res = res_[:6]
    thresh_res = res_[-5:]
    results.append(["ego"] + list(res))
    thresh_results.append(["ego"] + thresh_res[0] + thresh_res[1] + thresh_res[2] + thresh_res[3] + thresh_res[4])

    df_group = df_preds[df_preds['ego'] == 0]
    res_ = calculate_MORE_metrics(preds=df_group["probs"], labels=df_group["label"])
    res = res_[:6]
    thresh_res = res_[-5:]
    results.append(["noego"] + list(res))
    thresh_results.append(["noego"] + thresh_res[0] + thresh_res[1] + thresh_res[2] + thresh_res[3] + thresh_res[4])

    df_group = df_preds[df_preds['night'] == 1]
    res_ = calculate_MORE_metrics(preds=df_group["probs"], labels=df_group["label"])
    res = res_[:6]
    thresh_res = res_[-5:]
    results.append(["night"] + list(res))
    thresh_results.append(["night"] + thresh_res[0] + thresh_res[1] + thresh_res[2] + thresh_res[3] + thresh_res[4])

    df_group = df_preds[df_preds['night'] == 0]
    res_ = calculate_MORE_metrics(preds=df_group["probs"], labels=df_group["label"])
    res = res_[:6]
    thresh_res = res_[-5:]
    results.append(["day"] + list(res))
    thresh_results.append(["day"] + thresh_res[0] + thresh_res[1] + thresh_res[2] + thresh_res[3] + thresh_res[4])

    df = pd.DataFrame(results, columns=['group', 'acc', 'p', 'r', 'f1', 'map', 'auc'])
    df_thresh = pd.DataFrame(thresh_results, columns=["group"] + [f"mcc_{t:.2f}" for t in THRESHOLDS] + [f"p_{t:.2f}" for t in THRESHOLDS] + [f"r_{t:.2f}" for t in THRESHOLDS] + [f"acc_{t:.2f}" for t in THRESHOLDS] + [f"f1_{t:.2f}" for t in THRESHOLDS])
    df.to_csv(out_results_group)
    df_thresh.to_csv(out_metrics_thresholded)

    print(f"Done! Saved to: {out_results_group} and {out_metrics_thresholded}")


if __name__ == "__main__":
    DoTA_anno = "/datasets/DoTA/dataset/frame_level_anno_val.csv"
    DADA2K_anno = "/datasets/LOTVS/DADA/DADA2000/DADA2K_my_split/frame_level_anno_val.csv"

    logs = ["/VideoMAE_logs/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/201_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_bestmccauc"]
    
    anno_csv = DoTA_anno
    preds_dirs = logs

    for pred_dir in tqdm(preds_dirs):
        save_metrics(anno_csv=anno_csv, preds_dir=pred_dir)


