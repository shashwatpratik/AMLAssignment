import os
from pathlib import Path
import glob
from monai.data import ImageDataset
import nibabel as nib
import torch

def build_dataset(split_folder_path: str, type :str, transforms = None):
    valid_labels = ['CN', 'MCI', 'AD']
    images = []
    labels = []
    type_path = f'{split_folder_path}{type}.txt'
    data = read_lines_from_file(type_path)
    for f in data:
        img_array = nib.load(f).get_fdata()
        tensor_image = torch.from_numpy(img_array)
        check_flag = torch.isnan(tensor_image).any()
        if check_flag:
            continue
        filename = Path(f).stem
        label_name = filename.split('_')[-1]
        label = valid_labels.index(label_name)

        images.append(f)
        labels.append(label)
    dataset = ImageDataset(image_files=images, labels=labels, transform=transforms)
    return dataset, len(valid_labels)

def read_lines_from_file(file_path):
    data = []
    with open(file_path,mode='r') as file:
        for line in file:
            data.append(str(line).strip())
    return data