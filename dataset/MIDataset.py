from torch.utils.data.dataset import Dataset
import nibabel as nib
from pathlib import Path
import glob
import torch

class MIDataset(Dataset):
    def __init__(self, dataset_path, transforms = None):
        self.imgs = []
        self.transforms = transforms
        for f in glob.glob(f'{dataset_path}*\*'):
            img = nib.load(f)
            filename = Path(f).stem
            #if img.shape != (256, 256, 170):
            #    print(f'{filename} -> {img.shape}')
            #    continue
            img = img.get_fdata()
            label = filename.split('_')[-1]
            self.imgs.append((img, label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img, label = self.imgs[idx]
        img = torch.from_numpy(img)
        if self.transforms:
            img = self.transforms(img)
        return img, label

