import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, confusion_matrix
import xgboost as xgb


class ModelTrainer:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def _prepare_features_and_labels(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return X, y

    def split_data(self):
        X, y = self._prepare_features_and_labels()
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self, model):
        model.fit(self.X_train, self.y_train)
        return model

    def train_random_forest(self):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return self.train_model(model)

    def train_xgboost(self):
        model = xgb.XGBClassifier()
        return self.train_model(model)

    def hyperparameter_tuning(self, model, param_grid):
        """
        Tune the model's hyperparameters using cross-validation.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_

    def select_important_features(self, model, n_features=50):
        """
        Selects top n_features based on importance based on model
        """
        # Getting feature importance's
        importances = model.feature_importances_

        # Transform the feature importances into a readable format
        features = self.X_train.columns
        feature_importances = sorted(zip(importances, features), reverse=True)

        # Select the top n features
        top_n_features = [feature for importance, feature in feature_importances[:n_features]]

        # You can now use these top n features to transform your dataset
        self.X_train_selected = self.X_train[top_n_features]
        self.X_test_selected = self.X_test[top_n_features]

        # print(top_n_features)
        return top_n_features

    # Feature Importance
    def feature_importance_plot(self, model):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.X_train.shape[1]), importances[indices], align="center")
        plt.xticks(range(self.X_train.shape[1]), self.X_train.columns[indices], rotation=45, ha='right')
        plt.xlim([-1, self.X_train.shape[1]])
        plt.show()

    def remove_unimportant_features(self, model, threshold=0.01):
        importances = model.feature_importances_
        important_features_indices = np.where(importances > threshold)[0]

        # Always including the index of the 'Label' feature if it's not already included
        label_index = list(self.data.columns).index('Label')
        if label_index not in important_features_indices:
            important_features_indices = np.append(important_features_indices, label_index)

        # Getting the names of the important features
        important_features_names = [self.data.columns[i] for i in important_features_indices]

        # Keeping only the important features in the dataset
        self.data = self.data[important_features_names]
        print((self.data.columns.tolist()))
        return important_features_names

    @staticmethod
    def predict(model, X):
        return model.predict(X), model.predict_proba(X)[:, 1]

    @staticmethod
    def generate_report(y_true, y_pred, y_score, data_type):
        print(f"\n{data_type} Data Report\n")
        print(classification_report(y_true, y_pred))
        print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')
        print(f'F1 Score: {f1_score(y_true, y_pred):.4f}')
        print(f'Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}')
        print(f'{data_type} AUC: {roc_auc_score(y_true, y_score):.2f}')
        print("\n" + "-" * 50 + "\n")

    def evaluate_model(self, model):
        train_pred, train_score = self.predict(model, self.X_train)
        self.generate_report(self.y_train, train_pred, train_score, "Training")

        test_pred, test_score = self.predict(model, self.X_test)
        self.generate_report(self.y_test, test_pred, test_score, "Test")

        return train_pred, test_pred

    def cross_validate(self, model, cv=5):
        X, y = self._prepare_features_and_labels()
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"Cross-Validation Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean()}")
        print(f"No. of features: {len(self.data.columns)}")
