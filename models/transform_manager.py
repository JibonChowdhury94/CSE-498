import pandas as pd
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import cv2

from datasets.medical_image_dataset import MedicalImageDataset

class TransformManager:
    def __init__(self, image_folder, size, batch_size):
        self.image_folder = image_folder
        self.size = size
        self.batch_size = batch_size

    def prepare_data_for_fold(self, train_idx, val_idx):
        train_df = pd.read_csv("../train_fold.csv")
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]

        train_transform = A.Compose([
            # Geometric transforms
#             A.RandomResizedCrop(height=self.size, width=self.size, scale=(0.8, 1.0)),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),

            # Spatial transforms (especially helpful for medical images)
#             A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5, border_mode=cv2.BORDER_CONSTANT),
#             A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5, border_mode=cv2.BORDER_CONSTANT),
#             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5, border_mode=cv2.BORDER_CONSTANT),

            # Normalization and conversion to tensor
            A.Resize(height=self.size, width=self.size),
            A.CenterCrop(self.size, self.size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        
        valid_transform = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.CenterCrop(self.size, self.size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        

        # Initialize datasets and loaders with the transform
        train_dataset = MedicalImageDataset(train_fold_df, self.image_folder, transform=train_transform)
        val_dataset = MedicalImageDataset(val_fold_df, self.image_folder, transform=valid_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=32)

        return train_loader, val_loader
