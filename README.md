# Faulty Hardware

A comprehensive data preprocessing, visualization, and model training framework for fault detection in hardware systems.

## Description

This project comprises a systematic pipeline for handling, analyzing, and modeling data primarily aimed at fault detection in hardware systems i.e. Radio equipments. 
It includes functionalities ranging from data preprocessing steps such as missing value imputation and feature scaling, to model training and evaluation with various machine learning algorithms, 
and data visualization techniques to assess model performance and insights.
![Alt text](/images/Faulty%20Hardware%20UML.png "UML")

## Note
- The data set is confidential hence not provided.
- The suggested_features are also confidentiial and not provided.

## ML Pipeline

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
   - Facilitates training of machine learning models.
   - Implements hyperparameter tuning using GridSearchCV.
   - Provides functions for evaluating the trained model on training and test data.
   - Supports cross-validation for model evaluation.

4. **Base Model Interface** (`BaseModelInterface` ABC Class)
   - Template for creating machine learning models
   - Train and make predictions with the model using the provided training data.
   - Retrieve the underlying machine learning model.
   - Evaluate the performance of the model

5. **Random Forest Training** (`RandomForestModel` Class)
   - Model training using various algorithms using Random Forest.
   - Hyperparameter tuning using cross-validation.
   - Model evaluation and feature importance analysis.

6. **XGBoost Training** (`XGBoostModel` Class)
   - Model training using various algorithms using Random Forest.
   - Hyperparameter tuning using cross-validation.
   - Model evaluation and feature importance analysis.

7. **Neural Network Training** (`NeuralNetwork` Class)
   - Setting up the NN, define hidden layers, no. of Neurons.
   - Compile and train the NN.
   - Predict and evaluate the model.

![Alt text](/images/xgboost-test-scores.png "XGBoost test scores")

![Alt text](/images/xgboost-roc-auc.png "XGBoost ROC-AUC ")
## Usage

Here is a simple way to run the experiment:

```bash
from fault_hardware_model import run_experiment

file_path = "path_to_your_dataset.csv"
run_config = {
    "describe_data": False,
    "enable_SMOTE": True,
    "manual_feature_selection": True,  # For manual feature selection
    "apply_pca": False,
    "no_of_features": 100,
    "random_forest_basic": True,
    "random_forest_tuned": False,
    "random_forest_hyper_parameters": {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'bootstrap': [True, False]
    },
    "xgboost_basic": True,
    "xgboost_tuned": True,
    "xgboot_hyper_parameters": {
        'max_depth': [10],
        'learning_rate': [0.1],
        'n_estimators': [300],
        'subsample': [0.8],
        'colsample_bytree': [0.7],
        'reg_alpha': [0],
        'reg_lambda': [1.2]
    },
    "neural_net": False,
    "visualize": False  # Set this to False if you don't want visualizations
}

run_experiment(file_path, run_config)
```
