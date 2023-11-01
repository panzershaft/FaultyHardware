# Faulty Hardware

A comprehensive data preprocessing, visualization, and model training framework for fault detection in hardware systems.

## Description

This project comprises a systematic pipeline for handling, analyzing, and modeling data primarily aimed at fault detection in hardware systems i.e. Radio equipments. 
It includes functionalities ranging from data preprocessing steps such as missing value imputation and feature scaling, to model training and evaluation with various machine learning algorithms, 
and data visualization techniques to assess model performance and insights.

## Process Pipelines

1. **Data Preprocessing** (`DataPreprocessor` Class)
   - Loading data from a CSV file.
   - Dropping empty rows and columns.
   - Dropping specific columns.
   - Selecting features for model training.
   - Imputing missing values with specified strategies (mean by default).
   - Processing non-numeric features.
   - Dropping constant features.
   - Scaling features to have zero mean and unit variance.

2. **Data Visualization** (`DataVisualizer` Class)
   - Confusion Matrix plotting.
   - ROC Curve plotting.
   - Precision-Recall Curve plotting.

3. **Model Training** (`ModelTrainer` Class)
   - Splitting data into training and test sets.
   - Hyperparameter tuning using cross-validation.
   - Model training using various algorithms such as Random Forest and XGBoost.
   - Model evaluation and feature importance analysis.

## Usage

Here is a simple way to run the experiment:

```bash
from fault_hardware_model import run_experiment

file_path = "path_to_your_dataset.csv"
run_config = {
    "describe_data": True,
    "random_forest_basic": True,
    "random_forest_tuned": True,
    "xgboost_basic": True,
    "feature_importance": True,
    "visualize": True
}

run_experiment(file_path, run_config)
```

## Note
- The data set is confidential hence not provided.
- The suggested_features are also confidentiial and not provided.