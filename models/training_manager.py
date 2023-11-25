from tqdm.notebook import tqdm

class TrainingManager:
    def __init__(self, data_manager, transform_manager, model_manager, num_epochs):
        self.data_manager = data_manager
        self.transform_manager = transform_manager
        self.model_manager = model_manager
        self.num_epochs = num_epochs
#         self.n_splits = n_splits

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def run_training(self):
        splits = self.data_manager.get_splits()

        for fold, (train_idx, val_idx) in tqdm(enumerate(splits), total=len(splits), desc="Processing Folds"):
            
#             if(fold==1):
#                 print("Breaking Out of KFold")
#                 break;
                
            train_loader, val_loader = self.transform_manager.prepare_data_for_fold(train_idx, val_idx)
            train_loss, train_accuracy, val_loss, val_accuracy = self.model_manager.initialize_and_train_model(train_loader, val_loader, fold, self.num_epochs)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            