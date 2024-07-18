from src.config import suggested_features, FILE_MAPPING
# from src.models.neural_net_model import NeuralNetwork
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.training.model_trainer import ModelTrainer
from src.visualizing.data_visualizer import DataVisualizer


class ModelFactory:
    def __init__(self):
        self.builders = {}

    def register_model(self, model_name, model_class):
        self.builders[model_name] = model_class

    def create(self, model_name, **kwargs):
        model_class = self.builders.get(model_name)
        if not model_class:
            raise ValueError(f"{model_name} is not present")
        return model_class(**kwargs)

    def __str__(self):
        return f"Registered models: {list(self.builders.keys())}"


def initialize_model_factory():
    model_factory = ModelFactory()
    model_factory.register_model('RandomForest', RandomForestModel)
    model_factory.register_model('XGBoost', XGBoostModel)
    # model_factory.register_model('NeuralNetwork', NeuralNetwork)
    return model_factory


def run_model(model_name, model_trainer, param_grid, run_config):
    model = model_trainer
    print(f"\n{model_name:=^50}\n")
    if run_config.get('neural_net'):
        model.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], run_eagerly=True)
    model.hyperparameter_tuning(param_grid) if param_grid else model.train()
    model.get_model_cross_validation()
    model.get_model_evaluation()

    if run_config.get('visualize') and not run_config.get('neural_net'):
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
    preprocessor.process_data()

    if run_config.get("describe_data"):
        preprocessor.summarize_data()
    if run_config.get('manual_feature_selection'):
        (preprocessor.handle_outliers()
         .drop_constants()
         # .apply_pca(204)
         .manual_feature_select(run_config['no_of_features']))
    model_factory = initialize_model_factory()

    models_to_run = [
        ('Random Forest', 'random_forest_basic', 'RandomForest', None),
        (
        'Tuned Random Forest', 'random_forest_tuned', 'RandomForest', run_config.get("random_forest_hyper_parameters")),
        ('XGBoost', 'xgboost_basic', 'XGBoost', None),
        ('Tuned XGBoost', 'xgboost_tuned', 'XGBoost', run_config.get("xgboost_hyper_parameters")),
        ('Neural Network', 'neural_net', 'NeuralNetwork', None)
    ]

    for model_name, config_key, model_type, param_grid in models_to_run:
        if run_config.get(config_key):
            print(f"Running {model_name}")
            if model_type == 'NeuralNetwork':
                model_instance = model_factory.create(
                    model_type,
                    input_dim=run_config.get('no_of_features'),
                    activation='relu',
                    num_layers=2,
                    units_per_layer=[165, 330],
                    output_units=1,
                    output_activation='sigmoid'
                )
            else:
                model_instance = model_factory.create(model_type)
            model_trainer = ModelTrainer(model_instance, *preprocessor.address_data_imbalance(),
                                         *preprocessor.split_data())
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
    "xgboost_hyper_parameters": {
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

# Running the experiment
# run_experiment(FILE_MAPPING["file1"], run_config)
run_experiment(FILE_MAPPING["file2"], run_config)
# run_experiment(FILE_MAPPING["file3"], run_config)
