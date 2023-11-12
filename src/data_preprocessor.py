import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler


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
            print(f"Data loaded successfully from {self.filepath}")
        except FileNotFoundError:
            print(f"File not found: {self.filepath}")
        except Exception as e:
            print(f"Error occurred while loading the data: {e}")
        return self

    def drop_empty_rows_and_columns(self):
        self.data.dropna(axis=0, how='all', inplace=True)
        self.data.dropna(axis=1, how='all', inplace=True)
        return self

    def drop_column(self, column_name: str):
        if column_name in self.data.columns:
            self.data.drop(columns=[column_name], inplace=True)
            print(f"Column '{column_name}' dropped successfully.")
        else:
            print(f"Column '{column_name}' not found in the data.")
        return self

    def handle_outliers(self):
        z_scores = np.abs(stats.zscore(self.data.select_dtypes(include=[np.number])))
        threshold = 3  # Adjust based on domain knowledge
        outliers = (z_scores > threshold)
        median_values = self.data.median()
        self.data[outliers] = np.nan
        self.data.fillna(median_values, inplace=True)
        return self

    def select_features_for_model_training(self, suggested_features: list):
        final_features = [column for column in suggested_features if column in self.data.columns]
        self.data = self.data[final_features]
        return self

    def select_features(self, n_features=50):
        if 'Label' in self.data.columns:
            # Separate features and target variable
            X = self.data.drop('Label', axis=1)
            y = self.data['Label']

            selector = SelectKBest(f_classif, k=n_features)  # Adjust 'k' based on model performance
            X_new = selector.fit_transform(X, y)
            selected_columns = X.columns[selector.get_support()]
            self.data = pd.DataFrame(X_new, columns=selected_columns)
            self.data['Label'] = y
        else:
            print("Label column not found in the data.")
        return self

    def impute_missing_values(self, strategy='mean'):
        imputer = SimpleImputer(strategy=strategy)
        imputed_data = imputer.fit_transform(self.data)
        self.data = pd.DataFrame(imputed_data, columns=self.data.columns)
        return self

    def process_non_numerical_features(self):
        non_numerical_columns = self.data.select_dtypes(include=['object']).columns
        for column in non_numerical_columns:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
        return self

    def show_data_imbalance(self):
        class_counts = self.data['Label'].value_counts()
        print(class_counts)
        balance_ratio = class_counts.min() / class_counts.max()
        print(f'Balance Ratio: {balance_ratio:.2f}')
        return self

    def drop_constants(self):
        labels = self._get_labels_and_drop_if_exists()

        selector = VarianceThreshold()
        # Use the original column names when creating the new DataFrame
        self.data = pd.DataFrame(selector.fit_transform(self.data),
                                 columns=self.data.columns[selector.get_support(indices=True)])

        if labels is not None:
            self.data['Label'] = labels
        return self

    def scale_features(self, scaler_type='standard'):
        labels = self._get_labels_and_drop_if_exists()
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaler type")
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

        if labels is not None:
            self.data['Label'] = labels
        return self

    # Doesn't seem much useful for our case, performance degrades post PCA
    def apply_pca(self, n_components=10):
        """
        Apply PCA to reduce dimensionality.
        :param n_components: Number of components to keep.
        """
        labels = self._get_labels_and_drop_if_exists()
        pca = PCA(n_components=n_components)
        self.data = pd.DataFrame(pca.fit_transform(self.data), columns=self.data.columns[:n_components])
        if labels is not None:
            self.data['Label'] = labels
        return self

    def dimensions(self):
        return self.data.shape

    def describe(self):
        return self.data.describe()

    def summarize_data(self):
        if self.data is None:
            print("Data not loaded yet.")
            return self

        print(f"Data Summary for file: {self.filepath}\n")
        print(f"Data Dimensions: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"Total Missing Values: {self.data.isnull().sum().sum()}")
        print("Data Types:")
        print(self.data.dtypes)
        print("\nSummary Statistics:")
        print(self.data.describe())
        print(f"\nNumber of duplicate rows: {self.data.duplicated().sum()}")
