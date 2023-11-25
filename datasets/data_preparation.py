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
        # Using MultilabelStratifiedKFold for the initial split
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in mskf.split(self.df['SOPInstanceUID_with_png'], self.df[self.columns]):
            train_df = self.df.iloc[train_index]
            test_df = self.df.iloc[test_index]
            break  # only want the first split
            
        train_df.to_csv('train_fold.csv', index=False)
        test_df.to_csv('test_fold.csv', index=False)


#     def initial_split(self, n_splits=5, shuffle=True, random_state=42):
        
#         # Create a column that indicates the group (series) each slice belongs to
#         self.df['group'] = self.df['name'].factorize()[0]

#         mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

#         for fold_number, (train_group_idx, test_group_idx) in enumerate(mskf.split(self.df['group'], self.df[self.columns])):
#             # Convert group indices back to original DataFrame indices
#             train_idx = self.df[self.df['group'].isin(train_group_idx)].index
#             test_idx = self.df[self.df['group'].isin(test_group_idx)].index

#             # Extract train and test DataFrames based on the indices
#             train_df = self.df.iloc[train_idx]
#             test_df = self.df.iloc[test_idx]

#             # Save each fold to a separate CSV file
#             train_df.to_csv(f'train_fold_{fold_number}.csv', index=False)
#             test_df.to_csv(f'test_fold_{fold_number}.csv', index=False)

#     def get_splits(self, shuffle=True, random_state=42):
#             # Read the dataset
#             df = pd.read_csv('../train_fold.csv')

#             # Create a column that indicates the group (series) each slice belongs to
#             df['group'] = df['name'].factorize()[0]

#             # Initialize MultilabelStratifiedKFold
#             mskf = MultilabelStratifiedKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)

#             # Perform the split using the 'group' column to keep series together
#             # We also use self.columns as the target columns for stratification
#             splits = []
#             for train_group_idx, val_group_idx in mskf.split(df['group'], df[self.columns]):
#                 # Convert group indices back to original DataFrame indices
#                 train_idx = df[df['group'].isin(train_group_idx)].index
#                 val_idx = df[df['group'].isin(val_group_idx)].index

#                 # Append the indices to the splits
#                 splits.append((train_idx, val_idx))

#             return splits
        
    def get_splits(self, shuffle=True, random_state=42):
        mskf = MultilabelStratifiedKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        train_df = pd.read_csv('../train_fold.csv')
        return [(train_idx, val_idx) for train_idx, val_idx in mskf.split(train_df['SOPInstanceUID_with_png'], train_df[self.columns])]
