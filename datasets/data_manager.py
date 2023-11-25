from datasets.data_preparation import DataPreparation

class DataManager:
    def __init__(self, data_file, label_columns, n_splits):
        self.data_file = data_file
        self.label_columns = label_columns
        self.n_splits = n_splits
        
        self.data_prep = DataPreparation(self.data_file, self.label_columns, self.n_splits)
        self.data_prep.fill_na()

    def get_splits(self):
        return self.data_prep.get_splits()
