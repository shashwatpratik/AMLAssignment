import os
from pathlib import Path


class Settings:
    SEP = os.path.sep
    # For file based approch
    ROOT = f'{Path(__file__).resolve().parent.parent}'
    DATASET_PATH = f'{ROOT}{SEP}datasetx{SEP}'
    RAW_PATH = f'{ROOT}{SEP}raw{SEP}'
    SPLIT_PATH = f'{ROOT}{SEP}split{SEP}'
