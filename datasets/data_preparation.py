import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class DataPreparation:

    def __init__(self, csv_file, columns, n_splits):
        self.csv_file = csv_file
        self.columns = columns
        self.df = pd.read_csv(self.csv_file)
        self.n_splits = n_splits 

    def fill_na(self):
        for col in self.columns:
            self.df[col].fillna(0, inplace=True)

    def initial_split(self, n_splits=5, shuffle=False, random_state=None):
        
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in mskf.split(self.df['SOPInstanceUID_with_png'], self.df[self.columns]):
            train_df = self.df.iloc[train_index]
            test_df = self.df.iloc[test_index]
            break  # only want the first split
            
        train_df.to_csv('train_fold.csv', index=False)
        test_df.to_csv('test_fold.csv', index=False)
        
    def get_splits(self, shuffle=True, random_state=42):
        mskf = MultilabelStratifiedKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        train_df = pd.read_csv('../train_fold.csv')
        return [(train_idx, val_idx) for train_idx, val_idx in mskf.split(train_df['SOPInstanceUID_with_png'], train_df[self.columns])]
