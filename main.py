from src.config import suggested_features, FILE_MAPPING
from src.data_preprocessor import DataPreprocessor
from src.data_visualizer import DataVisualizer
from src.model_trainer import ModelTrainer


def run_model(model_name, train_func, extra_params, trainer, run_config):
    print(f"\n{model_name:=^50}\n")
    if extra_params:
        model = train_func(*extra_params)
    else:
        model = train_func()

    if run_config.get("manual_feature_selection"):
        print("\n--- Selecting Important Features ---\n")
        trainer.manual_feature_selection(model)

    trainer.cross_validate(model)
    trainer.evaluate_model(model)

    if run_config.get('visualize'):
        DataVisualizer.plot_confusion_matrix(trainer.y_test, model.predict(trainer.X_test),
                                             title=f'{model_name}-test')
        DataVisualizer.plot_roc_curve(trainer.y_test, model.predict_proba(trainer.X_test)[:, 1],
                                      title=f'{model_name}-test')
        DataVisualizer.plot_precision_recall_curve(trainer.y_test, model.predict_proba(trainer.X_test)[:, 1],
                                                   title=f'{model_name}-test')


def run_experiment(file_path, run_config):
    print(f'\n{"=" * 20} START OF {file_path} DATASET EVALUATION {"=" * 20}\n')

    # Data Preprocessing
    preprocessor = DataPreprocessor(file_path)
    preprocessor.load_data()
    if not run_config.get('No. of features'):
        preprocessor.select_features_for_model_training(suggested_features)
    preprocessor.drop_empty_rows_and_columns()
    preprocessor.process_non_numerical_features()
    preprocessor.impute_missing_values()
    preprocessor.drop_constants()
    preprocessor.scale_features()

    if run_config.get("describe_data"):
        # print(preprocessor.describe())
        preprocessor.summarize_data()

    # Model Training and Evaluation
    trainer = ModelTrainer(preprocessor.data, 'Label')

    models_to_run = [('Random Forest', 'random_forest_basic', trainer.train_random_forest, None),
                     ('Tuned Random Forest', 'random_forest_tuned', trainer.hyperparameter_tuning,
                      (trainer.train_random_forest(), run_config.get("random_forest_hyper_parameters"))),
                     ('XGBoost', 'xgboost_basic', trainer.train_xgboost, None)]

    for model_name, config_key, train_func, extra_params in models_to_run:
        if run_config.get(config_key):
            run_model(model_name, train_func, extra_params, trainer, run_config)

    print(f'\n{"=" * 20} END OF {file_path} DATASET EVALUATION {"=" * 20}\n')


# Configuration for the experiment
run_config = {
    "describe_data": False,
    "manual_feature_selection": True,
    "random_forest_basic": True,
    "random_forest_tuned": False,
    "random_forest_hyper_parameters": {
        # parameter set 1
        # 'n_estimators': [100, 200, 300],
        # 'max_depth': [None, 10, 20, 30],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4]
        # parameter set 2
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'bootstrap': [True, False]
    },
    "remove_unimportant_features": False,
    "xgboost_basic": False,
    "feature_importance": False,  # Set this to False if you don't want visualizations
    "visualize": False  # Set this to False if you don't want visualizations
}

# Running the experiment
run_experiment(FILE_MAPPING["file1"], run_config)
run_experiment(FILE_MAPPING["file2"], run_config)




