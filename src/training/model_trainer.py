from sklearn.model_selection import GridSearchCV

from src.models.base_model_interface import BaseModelInterface


class ModelTrainer:
    def __init__(self, model, X, y, X_train, X_test, y_train, y_test):
        self.model = model
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        self.model.train(self.X_train, self.y_train)

    def hyperparameter_tuning(self, param_grid):
        self.train()
        grid_search = GridSearchCV(estimator=self.model.get_model(), param_grid=param_grid, cv=5,
                                   scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        print("Best score: %0.3f" % grid_search.best_score_)
        # print("Best parameters set:", grid_search.best_params_)
        return grid_search.best_estimator_

    def get_model_evaluation(self):
        print(f"\n{'Training Data Report':-^50}\n")
        self.model.evaluate(self.X_train, self.y_train)
        print(f"\n{'Test Data Report':-^50}\n")
        self.model.evaluate(self.X_test, self.y_test)

    def get_model_cross_validation(self, cv=5):
        self.model.cross_validate(self.X, self.y, cv)
