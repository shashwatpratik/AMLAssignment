import os
import sys
from pathlib import Path
import glob
import shutil
import pandas as pd
import SimpleITK as sitk
from utils.loggers import Logger


class Settings:
    SEP = os.path.sep
    # For file based approch
    ROOT = f'{Path(__file__).resolve().parent.parent}'
    DATASET_PATH = f'{ROOT}{SEP}dataset{SEP}'
    RAW_PATH = f'{ROOT}{SEP}raw{SEP}'
    NAME_SEP = '_'


class FileConfig:
    Ext = '.nii'
    Replace = True


def get_dataset_metadata_pair(path: str = Settings.RAW_PATH):
    pair = {}
    datasets = tuple(str(f) for f in Path(path).iterdir() if f.is_dir())
    for dataset in datasets:
        name = Path(dataset).name
        dataset_name = str(name).replace(' ', '_')
        metafile = glob.glob(f'{path}{dataset_name}*.csv')[0]
        pair[name] = metafile
    return pair


def preprocessing(raw_path: str = Settings.RAW_PATH, dataset_path: str = Settings.DATASET_PATH):
    pairs = get_dataset_metadata_pair(raw_path)
    for pair in pairs.items():
        dataset, metadata = pair
        records = pd.read_csv(metadata, dtype=str, index_col='Image Data ID')
        imgs = tuple(img for img in glob.glob(f'{Settings.RAW_PATH}{dataset}/*/*/*/*/*/*'))
        assert len(records) == len(imgs), "Number of files and records mismatch"
        dest_folder = f'{dataset_path}{dataset}{Settings.SEP}'
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        counter = 0
        for img in imgs:
            img_idx = img.split(Settings.SEP)[-1].split(FileConfig.Ext)[-2].split(Settings.NAME_SEP)[-1]
            record = records.loc[img_idx]
            dest_file = f'{dest_folder}{record.Subject}_{img_idx}_{record.Group}{FileConfig.Ext}'
            print(f'{dataset} -> {record.Subject}')
            counter += 1
            print(f'{len(imgs) - counter}')
            inputImage = sitk.ReadImage(img, sitk.sitkFloat32)
            image = inputImage
            maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
            shrinkFactor = 1
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            numberFittingLevels = 4
            corrected_image = corrector.Execute(image, maskImage)
            log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
            corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
            sitk.WriteImage(corrected_image_full_resolution, dest_file)
            # shutil.copy2(img, dest_file)

    return True


if __name__ == "__main__":
    log_name = "log_Preprocess.txt"
    log_file = f'{Settings.ROOT}{Settings.SEP}{log_name}'
    sys.stdout = Logger(log_file)
    print(f'{log_file}')

    print(f'root:-> {Path(__file__).resolve().parent.parent}')
    print(f'dataset_path:-> {Settings.DATASET_PATH}')
    print(f'raw_path:-> {Settings.RAW_PATH}')
    preprocessing()
