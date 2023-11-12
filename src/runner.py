from src.config import suggested_features, FILE_MAPPING, sf2, selected_features
from src.data_preprocessor import DataPreprocessor
from src.data_visualizer import DataVisualizer
from src.model_trainer import ModelTrainer


def run_model(model_name, train_func, extra_params, trainer, run_config):
    print(f"\n{model_name:=^50}\n")

    model = train_func(*extra_params) if extra_params else train_func()

    # if run_config.get("manual_feature_selection"):
    #     trainer.select_important_features(model, run_config.get("no_of_features"))

    trainer.cross_validate(model)
    trainer.evaluate_model(model)

    if run_config.get('visualize'):
        predictions = model.predict(trainer.X_test)
        probabilities = model.predict_proba(trainer.X_test)[:, 1]
        DataVisualizer.plot_confusion_matrix(trainer.y_test, predictions,
                                             title=f'{model_name}-test')
        DataVisualizer.plot_roc_curve(trainer.y_test, probabilities,
                                      title=f'{model_name}-test')
        DataVisualizer.plot_precision_recall_curve(trainer.y_test, probabilities,
                                                   title=f'{model_name}-test')


def run_experiment(file_path, run_config):
    print(f'\n{"=" * 20} START OF {file_path} DATASET EVALUATION {"=" * 20}\n')

    # Data Preprocessing
    preprocessor = DataPreprocessor(file_path).load_data()
    preprocessor.select_features_for_model_training(suggested_features) \
        if not run_config.get('manual_feature_selection') else preprocessor.drop_column('ComplaintID')
    (preprocessor.drop_empty_rows_and_columns()
     .process_non_numerical_features()
     .show_data_imbalance()
     .handle_outliers()
     .impute_missing_values()
     .scale_features('robust')
     )

    if run_config.get("describe_data"):
        preprocessor.summarize_data()
    # Suggested_features: 96%
    # 50 features: 88% accuracy
    # 55 features: 90% accuracy
    # 60 features: 91% accuracy
    # 61 features: 87% accuracy
    # 69 features: 88% accuracy
    # 75 features: 91% accuracy
    # 100 features: 93% accuracy
    if run_config.get('manual_feature_selection'):
        (preprocessor.handle_outliers()
         .drop_constants()
         # .apply_pca(204)
         .select_features(100))

    # Model Training and Evaluation
    trainer = ModelTrainer(preprocessor.data, 'Label')

    models_to_run = [
        ('Random Forest', 'random_forest_basic', trainer.train_random_forest, None),
        ('Tuned Random Forest', 'random_forest_tuned', trainer.hyperparameter_tuning,
         (trainer.train_random_forest(), run_config.get("random_forest_hyper_parameters"))),
        ('XGBoost', 'xgboost_basic', trainer.train_xgboost, None),
        ('Tuned XGBoost', 'xgboost_tuned', trainer.hyperparameter_tuning,
         (trainer.train_xgboost(), run_config.get("xgboot_hyper_parameters")))
    ]

    for model_name, config_key, train_func, extra_params in models_to_run:
        if run_config.get(config_key):
            print(f"Running {model_name}")
            run_model(model_name, train_func, extra_params, trainer, run_config)

    print(f'\n{"=" * 20} END OF {file_path} DATASET EVALUATION {"=" * 20}\n')


# Configuration for the experiment
run_config = {
    "describe_data": False,
    "manual_feature_selection": False,  # will run Random forest with feature selection
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
    "xgboost_basic": False,
    "xgboost_tuned": False,
    "xgboot_hyper_parameters": {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 1],
        'colsample_bytree': [0.7, 0.8, 1],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.1, 1.2]
    },
    "visualize": False  # Set this to False if you don't want visualizations
}

# Running the experiment
run_experiment(FILE_MAPPING["file1"], run_config)
# run_experiment(FILE_MAPPING["file2"], run_config)
