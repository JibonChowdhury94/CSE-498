import torch
from tqdm.notebook import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_loss = []  # Adding lists to store loss and accuracy per epoch
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def _calculate_accuracy(self, outputs, labels):
        # Binarize the outputs using a threshold of 0.5
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct = (predicted == labels).float().sum()
        return (correct / (labels.size(0) * labels.size(1))).item()

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        for images, labels in tqdm(self.train_loader, total=len(self.train_loader), desc="Processing Training Folds"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_accuracy += self._calculate_accuracy(outputs, labels)
        average_accuracy = total_accuracy / len(self.train_loader)
        return total_loss, average_accuracy

    def validate(self):
        self.model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, total=len(self.val_loader), desc="Processing Validation Folds"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_val_loss += loss.item()
                total_val_accuracy += self._calculate_accuracy(outputs, labels)
        average_val_accuracy = total_val_accuracy / len(self.val_loader)
        return total_val_loss, average_val_accuracy

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()
            # Storing the results per epoch
            self.train_loss.append(train_loss)
            self.train_accuracy.append(train_accuracy)
            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_accuracy)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
