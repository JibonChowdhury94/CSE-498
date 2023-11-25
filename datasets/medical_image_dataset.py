from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class MedicalImageDataset(Dataset):

    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.dataframe.iloc[idx, 1]}"
        image = Image.open(img_name).convert("RGB")
        labels = torch.tensor(self.dataframe.iloc[idx, 2:].values.astype(float), dtype=torch.float32)

        if self.transform:
            image = self.transform(image=np.array(image))["image"]
#             image = self.transform(image)

        return image, labels
