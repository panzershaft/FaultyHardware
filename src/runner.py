from src.config import suggested_features, FILE_MAPPING
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.training.model_trainer import ModelTrainer
from src.visualizing.data_visualizer import DataVisualizer


def run_model(model_name, model_trainer, param_grid, run_config):
    model = model_trainer
    print(f"\n{model_name:=^50}\n")
    if param_grid:
        model.hyperparameter_tuning(param_grid)
    else:
        model.train()
    model.get_model_cross_validation()
    model.get_model_evaluation()

    if run_config.get('visualize'):
        predictions = model.model.get_model().predict(model.X_test)
        probabilities = model.model.get_model().predict_proba(model.X_test)[:, 1]
        DataVisualizer.plot_confusion_matrix(model.y_test, predictions,
                                             title=f'{model_name}-test')
        DataVisualizer.plot_roc_curve(model.y_test, probabilities,
                                      title=f'{model_name}-test')
        DataVisualizer.plot_precision_recall_curve(model.y_test, probabilities,
                                                   title=f'{model_name}-test')


def run_experiment(file_path, run_config):
    print(f'\n{"=" * 20} START OF {file_path} DATASET EVALUATION {"=" * 20}\n')

    # Data Preprocessing
    preprocessor = DataPreprocessor(file_path, run_config.get('enable_SMOTE'), 'Label')
    preprocessor.select_features_for_model_training(suggested_features) \
        if not run_config.get('manual_feature_selection') else preprocessor.drop_column('ComplaintID')
    (preprocessor.process_data())

    if run_config.get("describe_data"):
        preprocessor.summarize_data()
    if run_config.get('manual_feature_selection'):
        (preprocessor.handle_outliers()
         .drop_constants()
         # .apply_pca(204)
         .manual_feature_select(run_config['no_of_features']))
    X, y = preprocessor.address_data_imbalance()
    X_train, X_test, y_train, y_test = preprocessor.split_data()

    models_to_run = [
        ('Random Forest', 'random_forest_basic',
         ModelTrainer(RandomForestModel(), X, y, X_train, X_test, y_train, y_test), None),
        ('Tuned Random Forest', 'random_forest_tuned',
         ModelTrainer(RandomForestModel(), X, y, X_train, X_test, y_train, y_test),
         run_config.get("random_forest_hyper_parameters")),
        ('XGBoost', 'xgboost_basic', ModelTrainer(XGBoostModel(), X, y, X_train, X_test, y_train, y_test), None),
        ('Tuned XGBoost', 'xgboost_tuned', ModelTrainer(XGBoostModel(), X, y, X_train, X_test, y_train, y_test),
         run_config.get("xgboot_hyper_parameters")),
    ]

    for model_name, config_key, model_trainer, param_grid in models_to_run:
        if run_config.get(config_key):
            print(f"Running {model_name}")
            run_model(model_name, model_trainer, param_grid, run_config)

    print(f'\n{"=" * 20} END OF {file_path} DATASET EVALUATION {"=" * 20}\n')


# Configuration for the experiment
run_config = {
    "describe_data": False,
    "enable_SMOTE": True,
    "manual_feature_selection": True,  # For manual feature selection
    "apply_pca": False,
    "no_of_features": 100,
    "random_forest_basic": True,
    "random_forest_tuned": False,
    "random_forest_hyper_parameters": {
        # parameter set 1
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
    "visualize": True  # Set this to False if you don't want visualizations
}

# Running the experiment
# run_experiment(FILE_MAPPING["file1"], run_config)
run_experiment(FILE_MAPPING["file2"], run_config)
# run_experiment(FILE_MAPPING["file3"], run_config)
