import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, file_path, feature_names, target_col, 
                 target_mapping=None,  split_col=None, split_ratio=0.7, random_state=42):
        self.file_path = file_path
        self.feature_names = feature_names
        self.target_col = target_col
        self.target_mapping = target_mapping
        self.split_col = split_col
        self.split_ratio = split_ratio
        self.random_state = random_state

        self.data = self._load_data(file_path)

        self._validate_colomns()
        self.features = self.data[feature_names]
        self.target = self._process_target(self.data[target_col])

        self.train_data, self.test_data = self._split_data()
        self.X_train, self.y_train, self.X_test, self.y_test = self.get_train_test_data()

    def _load_data(self, file_path):
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or XLSX.")
        
    def _validate_colomns(self):
        missing_features = [col for col in self.feature_names if col not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in data: {missing_features}")
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' is not in the data.")
        
    def _process_target(self, target_series):
        if self.target_mapping:
            return target_series.map(self.target_mapping)
        elif target_series.dtype == 'object':
            unique_values = target_series.unique()
            if len(unique_values) == 2:
                return target_series.map({unique_values[0]: 0, unique_values[1]: 1})
            else:
                raise ValueError("Cannot infer target mapping from the data. Please specify `target_mapping`.")
        else:
            return target_series
        
    def _split_data(self):
        if self.split_col:
            if self.split_col not in self.data.columns:
                raise ValueError(f"Split column '{self.split_col}' is not in the data.")
            train_data = self.data[self.data[self.split_col] == 'train']
            test_data = self.data[self.data[self.split_col] == 'test']
        else:
            train_data, test_data = train_test_split(
                self.data, test_size=1 - self.split_ratio, random_state=self.random_state, stratify=self.target
            )
        return train_data, test_data
    
    def get_train_test_data(self):
        X_train = self.train_data[self.feature_names]
        y_train = self.train_data.index.map(self.target)
        X_test = self.test_data[self.feature_names]
        y_test = self.test_data.index.map(self.target)
        return X_train, y_train, X_test, y_test
    
if __name__ == '__main__':
    from .feature_manager import FeatureManager

    fm = FeatureManager().get_all_features()
    dataset = Dataset('data/processed/cleaned_data.csv', fm, 'group', {'R': 1, 'NR':0}, 'type')
    print(dataset.y_train)

