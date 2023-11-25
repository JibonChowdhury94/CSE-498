import pandas as pd
import torch
import os
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datasets.medical_image_dataset import MedicalImageDataset


class TestManager:
    def __init__(self, image_folder, data_file, label_columns, size, model, device, models_path, n_splits):
        self.image_folder = image_folder
        self.data_file = data_file
        self.label_columns = label_columns
        self.size = size
        self.model = model
        self.device = device
        self.models_path = models_path
        self.n_splits = n_splits

    def prepare_data_for_test(self):
        test_df = pd.read_csv(self.data_file)
        
        test_transform = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.CenterCrop(self.size, self.size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        test_dataset = MedicalImageDataset(test_df, self.image_folder, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=32)

        return test_loader
    
        
    def make_predictions(self):
            model = self.model(num_classes=len(self.label_columns)).to(self.device)
            test_loader = self.prepare_data_for_test()
            all_predictions = []
            all_true_labels = []
            all_raw_probabilities = []

            with torch.no_grad():
                for fold in tqdm(range(self.n_splits), desc="Processing Folds"):
                    model_path = os.path.join(self.models_path, f'fold_{fold + 1}_model.pth')
                    model.load_state_dict(torch.load(model_path))
                    model.eval()

                    fold_predictions = []
                    fold_true_labels = []
                    fold_raw_probabilities = []
                    
                    for images, labels in tqdm(test_loader, total=len(test_loader), desc="Processing Prediction"):
                        images = images.to(self.device)
                        outputs = model(images)
                        probabilities = torch.sigmoid(outputs).float().cpu().numpy()
                        predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                        fold_predictions.append(predicted)
                        fold_true_labels.append(labels.numpy())
                        fold_raw_probabilities.append(probabilities)

                    all_predictions.append(fold_predictions)
                    all_true_labels.append(fold_true_labels)
                    all_raw_probabilities.append(fold_raw_probabilities)

            return all_predictions, all_true_labels, all_raw_probabilities
        