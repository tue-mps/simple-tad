import os
import json
import zipfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
from deepspeed.ops.sparse_attention.trsrc import fname
from natsort import natsorted
from tqdm import tqdm
import matplotlib.ticker as plticker

mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'serif'

colors = {
    'green_p': '#006600',
    'green_l': '#bddbbd',
    'purple_p': '#8F4289',
    'purple_l': '#dfb9dc',
    'blue_p': '#134475',
    'blue_l': '#a7ccf1',
    'red_p': '#751320',
    'red_l': '#deb8bd',
    'curr_frame': '#71797e',
    'brown_grid': '#5c2b0a',
    'black': 'k',
}

FPS = 10


def get_clip_data(clip_name, df):
    clip_data = df[df['clip'] == clip_name].copy()
    return clip_data


def glue_with_margin_(imgs, margin=3, color=(0,0,0)):
    """
    imgs:   list of 3 arrays of shape (H, W, 3)
    margin: number of pixels between each image
    color:  RGB tuple to fill the gap
    """
    H, W, _ = imgs[0].shape
    # build a (H x margin x 3) block
    gap = np.zeros((H, margin, 3), dtype=imgs[0].dtype)
    gap[:] = color  # fill with e.g. black or white

    # now interleave: img0, gap, img1, gap, img2
    return np.hstack([imgs[0], gap, imgs[1], gap, imgs[2], gap, imgs[3]])


def glue_with_margin(imgs, timestamps, margin=3, color=(255,255,255),
                     font=cv2.FONT_HERSHEY_SIMPLEX,
                     font_scale=8, font_thickness=5,
                     text_color=(0,165,255), outline_color=(0,0,0)):
    """
    imgs: list of N arrays (H, W, 3)
    timestamps: list of N strings to draw on each img
    margin: px between imgs
    color: gap color
    """
    annotated = []
    for img, ts_ in zip(imgs, timestamps):
        ts = f"{ts_} s"
        h, w, _ = img.shape
        # compute text size & position
        ((text_w, text_h), _) = cv2.getTextSize(ts, font, font_scale, font_thickness)
        x = w - text_w - 60       # 10 px from right
        y = h - 30                # 10 px from bottom

        # draw outline (thicker black)
        cv2.putText(img, ts, (x, y), font, font_scale, outline_color,
                    thickness=font_thickness+2, lineType=cv2.LINE_AA)
        # draw main text
        cv2.putText(img, ts, (x, y), font, font_scale, text_color,
                    thickness=font_thickness, lineType=cv2.LINE_AA)

        annotated.append(img)

    # now build gaps
    H, W, _ = annotated[0].shape
    gap = np.zeros((H, margin, 3), dtype=annotated[0].dtype)
    gap[:] = color

    # interleave them all
    rows = []
    for im in annotated:
        rows.extend([im, gap])
    # drop last gap
    rows = rows[:-1]
    return np.hstack(rows)


def viz_three_and_curve(imgs,      # list of 4 H×W×3 uint8’s
                        probs1,    # 1D array of “no DAPT” risk scores
                        probs2,    # 1D array of “with DAPT” risk scores
                        labels,    # 1D boolean array (True=anomaly)
                        savepath=None):
    """
    imgs:   list of exactly 3 images (H×W×3 uint8)
    probs1: array of shape (T,)
    probs2: array of shape (T,)
    labels: bool array of shape (T,)  (True = anomaly)
    """
    # --- glue the three frames into one big H×(3W)×3 image ---
    ts = [0.1, 0.5, 0.9, 1.2]
    glued = glue_with_margin_(imgs, margin=3, color=(255,255,255))  # white gap

    # --- build figure + GridSpec: 2 rows, 1 column (top=glued, bottom=curve) ---
    fig = plt.figure(figsize=(15, 3.7))
    gs  = gridspec.GridSpec(2, 1,
                            height_ratios=[3, 2],
                            hspace=0.01)

    # top row: the single glued image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(cv2.cvtColor(glued, cv2.COLOR_BGR2RGB))  # convert BGR to RGB for matplotlib
    ax_img.axis("off")

    # TIMESTAMPS
    ts_list = [3.9, 4.2, 5.0, 9.0]  # PY
    #ts_list = [3.0, 4.4, 5.3, 5.5]  # Si
    n = len(ts_list)
    for i, ts in enumerate(ts_list):
        # we divide the horizontal axis [0,1] into n equal slots,
        # and put each label in the middle of its slot, at 10% up from bottom
        x = (i + 0.5)/n + 0.08
        y = 0.05
        ax_img.text(
            x, y, f"{ts-1.5:.1f} s",
            transform = ax_img.transAxes,
            fontsize=18,
            color="orange",
            weight="bold",
            ha="center", va="bottom",
            alpha=1.,
        )

    # bottom row: the combined risk‐score plot
    ax = fig.add_subplot(gs[1, 0])
    time = np.arange(len(probs1)) / 10.0  # assume 10 fps
    ymin, ymax = 0.0, 1.0

    ax.plot(time, probs1, label="w/o DAPT",   color="#8F1A55", lw=2)
    ax.plot(time, probs2, label="w/ DAPT", color="#1A788F", lw=2)

    # fill anomaly spans
    ax.fill_between(time, ymin, ymax, where=labels,
                    color="#9B398A", alpha=0.3,
                    transform=ax.get_xaxis_transform())

    # add horizontal line at y=0.5
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    # label anomaly
    anomaly_times = time[labels]
    if anomaly_times.size>0:
        mid_t = (anomaly_times.min() + anomaly_times.max()) / 2
        # place the text just above y=0.5
        ax.text(
            3.5, 0.52, "ground truth \nanomaly window",  # 3.3 for PY, 3.5 for 
            ha="center", va="bottom",
            fontsize=12, color="#353535",
            bbox=dict(
                facecolor="none",   # or whatever base color you like
                alpha=0.8,           # make it semi-transparent
                edgecolor="none",    # no border
                pad=2                # little padding around text
            )
        )

    # limits
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(ymin-0.015, ymax+0.02)

    # set x‐ticks every 0.2 s (or 0.1)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(0.3))
    # ax.xaxis.set_major_locator(plticker.MultipleLocator(0.1))  # for 0.1s

    # axis labels with larger font
    ax.set_xlabel("Time (s)",     fontsize=12)
    ax.set_ylabel("Risk score",   fontsize=14)

    # tick label font size
    ax.tick_params(axis='both', which='major', labelsize=10)

    # legend with its own font‐size
    ax.legend(loc="upper left", frameon=False, fontsize=12)  # "upper right" for PY

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()

"""
This script produces the visualization plots as shown in the paper
"""


ckpt = 3
tag = "_train"  # "_train
version = "small"
# small
predictions1 = "/VideoMAE_logs/train_logs/finetune/DoTA_orig/vm1-small_dota_lr1e3_b56x1_dsampl1val1_ld06_aam6n3/eval_DoTA_ckpt_bestmccauc/predictions.csv"
predictions2 = "/VideoMAE_logs/train_logs/finetune/DoTA_orig/vm1-small-dapt_dota_lr1e3_b56x1_dsampl1val1_ld06_aam6n3/eval_DoTA_ckpt_bestmccauc/predictions.csv"

out_folder = f"/plots"

video_dir = "/datasets/DoTA/frames"
seq_length = 16

# Here are 4 selected images from the video clip
frame_folder = "/VideoMAE_logs/visualizations/PYL3JcSsS6o_004036_3"
clip_names = ["PYL3JcSsS6o_004036"]  # Sihe6aeyLHg_000602, PYL3JcSsS6o_004036

clip_name = clip_names[0]

df = pd.read_csv(predictions1)
clip_data1 = get_clip_data(clip_name, df)
logits1 = clip_data1[["logits_safe", "logits_risk"]].values  # Shape (N, 2)
logits_tensor1 = torch.tensor(logits1, dtype=torch.float32)
probabilities1 = torch.nn.functional.softmax(logits_tensor1, dim=1).numpy()[:, 1]

labels1 = list(clip_data1["label"].values)

df = pd.read_csv(predictions2)
clip_data2 = get_clip_data(clip_name, df)
logits2 = clip_data2[["logits_safe", "logits_risk"]].values  # Shape (N, 2)
logits_tensor2 = torch.tensor(logits2, dtype=torch.float32)
probabilities2 = torch.nn.functional.softmax(logits_tensor2, dim=1).numpy()[:, 1]

labels2 = list(clip_data2["label"].values)

assert labels1 == labels2, "Labels mismatch between two predictions files!"

labels = labels1

files = os.listdir(frame_folder)
files = natsorted([f for f in files if f.endswith(".jpg")])
imgs = [cv2.imread(os.path.join(frame_folder, f)) for f in files]
print(len(imgs))

viz_three_and_curve(imgs, probabilities1, probabilities2, labels,
                savepath=f"/VideoMAE_logs/visualizations/{clip_name}_viz.pdf")

print("Saved! ", clip_name)

