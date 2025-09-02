import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, matthews_corrcoef, auc

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


THRESHOLDS = np.arange(0.00, 1.001, 0.01).tolist()
cat_codes = ["ST", "AH", "LA", "OC", "TC", "VP", "VO", "OO", "UK"]


def get_mcc_metrics(labels, probs):
    probs = np.array(probs)
    mcc_t_vals = []
    
    for t in THRESHOLDS:
        binary_preds_t = (probs >= t).astype(int)
        mcc_val = matthews_corrcoef(labels, binary_preds_t)
        mcc_t_vals.append(mcc_val)

    mcc_max = max(mcc_t_vals)
    mcc_max_idx = mcc_t_vals.index(mcc_max)
    idx_05 = THRESHOLDS.index(0.5)
    mcc_05 = mcc_t_vals[idx_05]
    mcc_auc = auc(THRESHOLDS, mcc_t_vals)
    
    return mcc_auc, mcc_05


def show_metrics(anno_csv, preds_dir):
    pred_csv = os.path.join(preds_dir, "predictions.csv")
    out_file = os.path.join(preds_dir, "group_metrics.txt")
    results = []

    # ——— check for existing output ———
    if os.path.exists(out_file):
        resp = input(f"⚠️  Output file already exists. \n{out_file}\n\t❓ Overwrite? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("🛑  Stop! Do not overwrite! Existing metrics file left intact.")
            return
        else:
            print("▶️  Will overwrite the output file. Continue running...")
    # —————————————

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
    
    # Print some info
    results.append(f"Anno file: {anno_csv}")
    results.append(f"Preds file: {pred_csv}")

    annot_subset = df_anno[['clip', 'filename', 'ego', 'night', 'cat', 'clip_lvl_cat', 'clip_lvl_ego']]
    df_preds = df_preds.merge(annot_subset, on=['clip', 'filename'], how='left')

    if df_preds.isnull().values.any():
        num_rows = df_preds.shape[0]
        num_rows_with_nan = df_preds.isnull().any(axis=1).sum()
        percent = num_rows_with_nan / num_rows
        results.append(f"There are {percent*100:.2f}% missing values in the DataFrame while normal max percent is {ok_percent}.")
        if percent < ok_percent:
            results.append("It's okay. Remove invalid rows and proceed...")
            df_preds = df_preds.dropna()
        else:
            results.append("It's not okay. Halt.")
            exit(0)
    else:
        results.append("No missing values found!")

    del annot_subset
    del df_anno

    # CALCULATE METRICS

    df_preds["cat"] = df_preds["cat"].astype(int)
    df_preds["ego"] = df_preds["ego"].astype(int)
    df_preds["night"] = df_preds["night"].astype(int)
    #unique_categories = sorted(df_preds["cat"].unique().tolist())
    unique_categories = set(df_preds["clip_lvl_cat"].unique())
    assert unique_categories == set(cat_codes)
    unique_categories = cat_codes

    results.append("===========================================================")
    results.append("  General")
    results.append("-----------------------------------------------------------")

    clips = set(df_preds["clip"].to_list())
    auroc = roc_auc_score(df_preds["label"], df_preds["probs"])
    mcc_auc, mcc_05 = get_mcc_metrics(df_preds["label"], df_preds["probs"])
    results.append(f"TOTAL")
    results.append(f"\tlen: {len(clips)} | auroc: {100*auroc:.1f} | aucmcc: {100*mcc_auc:.1f} | mcc05: {100*mcc_05:.1f}")

    results.append("===========================================================")
    results.append("  General by categories")
    results.append("-----------------------------------------------------------")

    for uc in unique_categories:
        df_cat = df_preds[df_preds['cat'] == uc]
        cat_clips = set(df_cat["clip"].to_list())
        df_cat = df_preds[df_preds['clip'].isin(cat_clips)]
        df_cat = df_preds[df_preds['clip_lvl_cat'] == uc]  # 
        cat_clips = set(df_cat["clip"].to_list())  #
        auroc = roc_auc_score(df_cat["label"], df_cat["probs"])
        mcc_auc, mcc_05 = get_mcc_metrics(df_cat["label"], df_cat["probs"])
        results.append(f"category {uc}")
        results.append(f"\tlen: {len(cat_clips)} | auroc: {100*auroc:.1f} | aucmcc: {100*mcc_auc:.1f} | mcc05: {100*mcc_05:.1f}")

    results.append("===========================================================")
    results.append("  EGO by categories")
    results.append("-----------------------------------------------------------")

    df_group = df_preds[df_preds['clip_lvl_ego'] == True]
    ego_clips = set(df_group["clip"].to_list())
    auroc = roc_auc_score(df_group["label"], df_group["probs"])
    mcc_auc, mcc_05 = get_mcc_metrics(df_group["label"], df_group["probs"])
    results.append(f"GROUP EGO")
    results.append(f"\tlen: {len(ego_clips)} | auroc: {100*auroc:.1f} | aucmcc: {100*mcc_auc:.1f} | mcc05: {100*mcc_05:.1f}")
    results.append("-----------------------------------------------------------")

    for uc in unique_categories:
        df_cat = df_group[df_group['cat'] == uc]
        cat_clips = set(df_cat["clip"].to_list())
        df_cat = df_group[df_group['clip'].isin(cat_clips)]
        df_cat = df_group[df_group['clip_lvl_cat'] == uc]  # 
        cat_clips = set(df_cat["clip"].to_list())  #
        auroc = roc_auc_score(df_cat["label"], df_cat["probs"])
        mcc_auc, mcc_05 = get_mcc_metrics(df_cat["label"], df_cat["probs"])
        results.append(f"category {uc}")
        results.append(f"\tlen: {len(cat_clips)} | auroc: {100*auroc:.1f} | aucmcc: {100*mcc_auc:.1f} | mcc05: {100*mcc_05:.1f}")


    results.append("===========================================================")
    results.append("  NON-EGO by categories")
    results.append("-----------------------------------------------------------")

    df_group = df_preds[df_preds['clip_lvl_ego'] == False]
    nonego_clips = set(df_group["clip"].to_list())
    auroc = roc_auc_score(df_group["label"], df_group["probs"])
    mcc_auc, mcc_05 = get_mcc_metrics(df_group["label"], df_group["probs"])
    results.append(f"GROUP NON-EGO")
    results.append(f"\tlen: {len(nonego_clips)} | auroc: {100*auroc:.1f} | aucmcc: {100*mcc_auc:.1f} | mcc05: {100*mcc_05:.1f}")
    results.append("-----------------------------------------------------------")

    for uc in unique_categories:
        df_cat = df_group[df_group['cat'] == uc]
        cat_clips = set(df_cat["clip"].to_list())
        df_cat = df_group[df_group['clip'].isin(cat_clips)]
        df_cat = df_group[df_group['clip_lvl_cat'] == uc]  # 
        cat_clips = set(df_cat["clip"].to_list())  #
        auroc = roc_auc_score(df_cat["label"], df_cat["probs"])
        mcc_auc, mcc_05 = get_mcc_metrics(df_cat["label"], df_cat["probs"])
        results.append(f"category {uc}")
        results.append(f"\tlen: {len(cat_clips)} | auroc: {100*auroc:.1f} | aucmcc: {100*mcc_auc:.1f} | mcc05: {100*mcc_05:.1f}")

    results = "\n".join(results)
    with open(out_file, "w") as f:
        f.write(results)

    print(f"Saved to {out_file}", end="\n\n")
    print(results)


if __name__ == "__main__":
    DoTA_anno = "/datasets/DoTA/dataset/frame_level_anno_val.csv"

    log_dir = "/VideoMAE_logs/train_logs/ft_after_pretrain/bl2/205_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/mass_eval/_mccbest_eval_DoTA_ckpt_45"

    show_metrics(anno_csv=DoTA_anno, preds_dir=log_dir)


