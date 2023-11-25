import os
import torch
import torch.nn as nn
import torch.optim as optim
# import sys
# sys.path.append('../')

from datasets.medical_image_dataset import MedicalImageDataset
from models.trainer import Trainer
from models.loss import FocalLoss

class ModelManager:
    def __init__(self, label_columns, device, save_path, model, learning_rate, weight_decay, optimizer_type):
        self.label_columns = label_columns
        self.device = device
        self.save_path = save_path
        self.num_classes = len(self.label_columns)
        self.learning_rate = learning_rate
        self.model = model
        self.weight_decay = weight_decay 
        self.optimizer_type = optimizer_type
        

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _initialize_model(self):
        model = self.model(num_classes=self.num_classes).to(self.device)
        return model

    def _initialize_loss(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def _initialize_optimizer(self, model):
        optimizer_classes = {
        'Adam': optim.Adam,
        'RAdam': optim.RAdam
        }
        
        if self.optimizer_type not in optimizer_classes:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        optimizer_class = optimizer_classes[self.optimizer_type]
        optimizer = optimizer_class(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


    def save_model(self, model, fold):
        model_filename = os.path.join(self.save_path, f'fold_{fold + 1}_model.pth')
        torch.save(model.state_dict(), model_filename)

    def initialize_and_train_model(self, train_loader, val_loader, fold, num_epochs):
        model = self._initialize_model()
        criterion = self._initialize_loss()
        optimizer = self._initialize_optimizer(model)

        trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, device=self.device)
        trainer.fit(num_epochs)

        # Save the trained model
        self.save_model(trainer.model, fold)

        return trainer.train_loss, trainer.train_accuracy, trainer.val_loss, trainer.val_accuracy
