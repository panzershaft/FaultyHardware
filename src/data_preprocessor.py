import pandas as pd
import logging

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """
        Load data from a CSV file and handle exceptions if the file is not found or another error occurs.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            logging.info(f"Data loaded successfully from {self.filepath}")
        except FileNotFoundError:
            logging.error(f"File not found: {self.filepath}")
        except Exception as e:
            logging.warning(f"Error occurred while loading the data: {e}")

    def drop_empty_rows_and_columns(self):
        """
        Drop rows and columns that are entirely NA.
        """
        self.data.dropna(axis=0, how='all', inplace=True)
        self.data.dropna(axis=1, how='all', inplace=True)

    def drop_column(self, column_name):
        """
        Drop a specific column from the data.
        """
        if column_name in self.data.columns:
            self.data.drop(columns=[column_name], inplace=True)
            logging.info(f"Column '{column_name}' dropped successfully.")
        else:
            logging.warning(f"Column '{column_name}' not found in the data.")

    def select_features_for_model_training(self, suggested_features):
        """
        Select only the specified features/columns from the data.
        """
        final_features = [column for column in suggested_features if column in self.data.columns]
        self.data = self.data[final_features]

    def impute_missing_values(self, strategy='mean'):
        """
        Impute missing values in the data using a specified strategy (mean by default).
        """
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy)
        imputed_data = imputer.fit_transform(self.data)
        self.data = pd.DataFrame(imputed_data, columns=self.data.columns)  # Update the dataframe with imputed values.

    def process_non_numerical_features(self):
        """
        Convert non-numeric columns to numeric by coercing errors.
        """
        non_numerical_columns = self.data.select_dtypes(include=['object']).columns
        for column in non_numerical_columns:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

    def drop_constants(self):
        """
        Remove constant columns from the data except for the 'Label' column.
        """
        # Separating the 'Label' column from the dataset
        labels = None
        if 'Label' in self.data.columns:
            labels = self.data.pop('Label')

        # Applying VarianceThreshold to remove constant columns
        selector = VarianceThreshold()
        self.data = pd.DataFrame(selector.fit_transform(self.data))

        # Adding back the 'Label' column
        if labels is not None:
            self.data['Label'] = labels

    def scale_features(self):
        """
        Scale the features to have zero mean and unit variance, excluding the 'Label' column.
        """
        labels = None
        if 'Label' in self.data.columns:
            labels = self.data.pop('Label')

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(scaled_data, columns=self.data.columns)

        if labels is not None:
            self.data['Label'] = labels

    def dimensions(self):
        """
        Return the dimensions of the data.
        """
        return self.data.shape

    def describe(self):
        return self.data.describe()

    def summarize_data(self):
        """
        Summarizes various aspects of the data to give an overview of its content and structure.
        """
        # Check if data is loaded
        if self.data is None:
            print("Data not loaded yet.")
            return

        # Display basic info about the data
        print(f"Data Summary for file: {self.filepath}\n")
        print(f"Data Dimensions: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

        # Display presence of missing values
        missing_values = self.data.isnull().sum().sum()
        print(f"Total Missing Values: {missing_values}")

        # Display data types of columns
        print("Data Types:")
        print(self.data.dtypes)

        # Display summary statistics
        print("\nSummary Statistics:")
        print(self.data.describe())

        # Checking duplicates
        duplicates = self.data.duplicated().sum()
        print(f"\nNumber of duplicate rows: {duplicates}")
