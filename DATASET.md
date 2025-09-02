# Data Preparation

⚠️ Please note that due to the limit on the number of files, we packed extracted frames of each video into a .zip file (`images.zip`) for datasets DoTA, DADA-2000 and CAP-DATA. \
You can use `data_tools/frames2zip.py` to pack frames. \
As an alternative, you can replace the function that reads frames. For the DoTA dataset, it's in `dota.py`, class `FrameClsDataset_DoTA`, method `load_images`, and similarly with all the other datasets.

Apart from that, we use original structures of each dataset.

## Fine-tuning

### DoTA

! Note that we used .zip files with this dataset

Use original instructions to download and prepare the dataset: [DoTA repo](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly/tree/master).
You will have the following file structure: 
``` 
DoTA
├ dataset
└ frames 
```

To finetune on the half of the dataset, use the following split: [DoTA/dataset/half_train_split.txt](https://tuenl-my.sharepoint.com/:t:/g/personal/s_orlova_tue_nl/EaBIJyppHVhNo9rpk0u7ghkBEXklBuAm8cU8mM4UjGQmjA?e=oQqHES), or you can generate it yourself: `data_tools/dota/halfsplit.py`

To calculate metrics from saved predictions, use the following annotation file: [DoTA/dataset/frame_level_anno_val.csv](https://tuenl-my.sharepoint.com/:x:/g/personal/s_orlova_tue_nl/EVo9qdf7YF9Jkn27b4s4JuYBPHb-JE1eBplBEjSKwyETYg?e=NWoIKH), or you can generate it yourself: `data_tools/dota/anno_for_predictions.py`

### DADA-2000

! Note that we used .zip files with this dataset

Download [DADA-2000](https://github.com/JWFangit/LOTVS-DADA). We downloaded the full benchmark with already extracted frames, then reformat split files and annotations for easier integration with our code. 
Download [annotations](https://tuenl-my.sharepoint.com/:u:/g/personal/s_orlova_tue_nl/EaQQU0yyDrpGoPsE1c5sGeQBnpKHDVWjcpmadUHweZ5yCw?e=pGQQrn) and extract them into the DADA-2000 folder.

You should get the structure like this: 

```
DADA2000
├ annotation
├ DADA2K_my_split 
└ frames 
    ├ 1
        ├ 001
        ├ ... 
        └ 053 
    ├ ...
    └ 61
```

To finetune on the half of the dataset, use the following split: [DADA2000/DADA2K_my_split/half_training.txt](https://tuenl-my.sharepoint.com/:t:/g/personal/s_orlova_tue_nl/EaBIJyppHVhNo9rpk0u7ghkBEXklBuAm8cU8mM4UjGQmjA?e=oQqHES), or you can generate it yourself: `data_tools/dada/halfsplit.py`

To calculate metrics from saved predictions, use the following annotation file: [DADA2000/DADA2K_my_split/frame_level_anno_val.csv](https://tuenl-my.sharepoint.com/:x:/g/personal/s_orlova_tue_nl/EVo9qdf7YF9Jkn27b4s4JuYBPHb-JE1eBplBEjSKwyETYg?e=NWoIKH), or you can generate it yourself: `data_tools/dada/anno_for_predictions.py`

## DAPT - Domain-Adaptive Pre-training

For DAPT, we don't need any annotations. However, we prepare split files, and sometimes sample lists. 

### BDD100K

BDD100K has a large amount of frames, so indexing the whole dataset and preparing a list of samples is time-consuming. 
Therefore, we do it beforehand and save .pkl files that are then read by our dataloader. Download them from here and unzip as `bdd100k_splits`: [bdd100k_splits](https://tuenl-my.sharepoint.com/:u:/g/personal/s_orlova_tue_nl/EcrEwflPNFBIihmES017NxUBXrK9YolvIRXbLUu6HavtwA?e=dC2njr)

Provide the path to this folder in `datasets_frame.py, line 121`

### CAP-DATA

! Note that we used .zip files with this dataset

Prepared sample lists and annotations in our format: [cap_data.zip](https://tuenl-my.sharepoint.com/:u:/g/personal/s_orlova_tue_nl/Ed9zqXpaI8NDm9PmRci0zoEBlmY6P2OMTZ_3OJCpHPaSyw?e=pCiKMy). \
Unzip into CAP-DATA folder, so you have the structure identical to DADA-2000:

```
CAP-DATA
├ annotation
├ CAPDATA_my_split 
└ frames 
    ├ 1
        ├ 001537
        ├ ... 
        └ 014434 
    ├ ...
    └ 62
```

### Kinetics

Just download the original dataset from [Kinetics repo](https://github.com/cvdfoundation/kinetics-dataset). You don't need to do anything else.

## Refined DoTA dataset (TBD)

⚠️ Please note that we didn't see any significant improvement with these refined annotations. We provide it for reference only.

TBD
