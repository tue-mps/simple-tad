import pandas as pd


if __name__ == "__main__":
    anno_file = '/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/annotation/cap_text_annotations.xls'
    out_anno_file = '/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/annotation/cap_text_annotations.csv'

    df = pd.read_excel(anno_file, sheet_name=None)
    df = df["annotation file"]
    df.to_csv(out_anno_file, index=None)


