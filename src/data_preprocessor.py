import pandas as pd
import logging
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def _get_labels_and_drop_if_exists(self):
        labels = None
        if 'Label' in self.data.columns:
            labels = self.data.pop('Label')
        return labels

    def load_data(self):
        try:
            self.data = pd.read_csv(self.filepath)
            logging.info(f"Data loaded successfully from {self.filepath}")
        except FileNotFoundError:
            logging.error(f"File not found: {self.filepath}")
        except Exception as e:
            logging.warning(f"Error occurred while loading the data: {e}")

    def drop_empty_rows_and_columns(self):
        self.data.dropna(axis=0, how='all', inplace=True)
        self.data.dropna(axis=1, how='all', inplace=True)

    def drop_column(self, column_name: str):
        if column_name in self.data.columns:
            self.data.drop(columns=[column_name], inplace=True)
            print(f"Column '{column_name}' dropped successfully.")
        else:
            print(f"Column '{column_name}' not found in the data.")

    def select_features_for_model_training(self, suggested_features: list):
        final_features = [column for column in suggested_features if column in self.data.columns]
        self.data = self.data[final_features]

    def impute_missing_values(self, strategy='mean'):
        imputer = SimpleImputer(strategy=strategy)
        imputed_data = imputer.fit_transform(self.data)
        self.data = pd.DataFrame(imputed_data, columns=self.data.columns)

    def process_non_numerical_features(self):
        non_numerical_columns = self.data.select_dtypes(include=['object']).columns
        for column in non_numerical_columns:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

    def drop_constants(self):
        labels = self._get_labels_and_drop_if_exists()

        selector = VarianceThreshold()
        self.data = pd.DataFrame(selector.fit_transform(self.data))

        if labels is not None:
            self.data['Label'] = labels

    def scale_features(self):
        labels = self._get_labels_and_drop_if_exists()

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(scaled_data, columns=self.data.columns)

        if labels is not None:
            self.data['Label'] = labels

    def dimensions(self):
        return self.data.shape

    def describe(self):
        return self.data.describe()

    def summarize_data(self):
        if self.data is None:
            print("Data not loaded yet.")
            return

        print(f"Data Summary for file: {self.filepath}\n")
        print(f"Data Dimensions: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"Total Missing Values: {self.data.isnull().sum().sum()}")
        print("Data Types:")
        print(self.data.dtypes)
        print("\nSummary Statistics:")
        print(self.data.describe())
        print(f"\nNumber of duplicate rows: {self.data.duplicated().sum()}")