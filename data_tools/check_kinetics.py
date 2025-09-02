import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def _check_clip_duration(task):
    """
    Worker: given the minimal info to locate & inspect one clip,
    return its DataFrame index if duration >= min_duration, else None.
    """
    idx, ytid, lbl, t1_f, t2_f, data_root, subset_folder, video_ext, min_duration = task
    from decord import VideoReader, cpu

    # build the filename from frame-number strings
    t1 = str(int(t1_f)).zfill(6)
    t2 = str(int(t2_f)).zfill(6)
    clip_name = f"{ytid}_{t1}_{t2}{video_ext}"
    clip_path = os.path.join(data_root, subset_folder, lbl, clip_name)

    if not os.path.isfile(clip_path):
        # missing file → drop
        return None

    try:
        vr = VideoReader(clip_path, ctx=cpu(0), num_threads=1)
        duration = len(vr) / vr.get_avg_fps()
    except Exception:
        # any read error → drop
        return None

    return idx if duration >= min_duration else None


def filter_long_videos_parallel(orig_csv: str,
                                data_root: str,
                                subset_folder: str,
                                video_ext: str = ".mp4",
                                min_duration: float = 1.6,
                                n_workers: int = 8):
    # 1) load original CSV
    df = pd.read_csv(orig_csv)
    base, ext = os.path.splitext(orig_csv)
    out_csv = "/datasets/kinetics-700/train_long.csv"

    # 2) build task list
    tasks = []
    for idx, row in df.iterrows():
        tasks.append((
            idx,
            row["youtube_id"],
            row["label"],
            row["time_start"],
            row["time_end"],
            data_root,
            subset_folder,
            video_ext,
            min_duration
        ))

    # 3) parallel duration checks
    keep_idx = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for result in tqdm(exe.map(_check_clip_duration, tasks),
                           total=len(tasks),
                           desc="Filtering by duration"):
            if result is not None:
                keep_idx.append(result)

    # 4) slice & save
    df_long = df.loc[keep_idx].reset_index(drop=True)
    df_long.to_csv(out_csv, index=False)
    print(f"Kept {len(df_long)}/{len(df)} clips ≥{min_duration}s → {out_csv}")


if __name__ == "__main__":
    DATA_ROOT    = "/scratch-nvme/ml-datasets/kinetics/k700-2020"
    SUBSET       = "train"   # e.g. the subfolder under DATA_ROOT containing your mp4s
    ORIGINAL_CSV = os.path.join(DATA_ROOT, "annotations", "train.csv")

    filter_long_videos_parallel(
        orig_csv=ORIGINAL_CSV,
        data_root=DATA_ROOT,
        subset_folder=SUBSET,
        video_ext=".mp4",
        min_duration=1.6,
        n_workers=16    # or however many CPU cores you want to devote
    )
