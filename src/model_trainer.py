import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb


class ModelTrainer:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def split_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def hyperparameter_tuning(self, model, param_grid):
        """
        Tune the model's hyperparameters using cross-validation.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        return best_model

    def train_random_forest(self):
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        return model

    def train_xgboost(self):
        model = xgb.XGBClassifier()
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate_model(self, model):
        train_predictions = model.predict(self.X_train)
        print(f"\n{'Training Data Report':-^50}\n")
        print(classification_report(self.y_train, train_predictions))
        print(f"Accuracy: {accuracy_score(self.y_train, train_predictions)}\n")

        # Evaluating on Test Data
        test_predictions = model.predict(self.X_test)
        print(f"\n{'Test Data Report':-^50}\n")
        print(classification_report(self.y_test, test_predictions))
        print(f"Accuracy: {accuracy_score(self.y_test, test_predictions)}")
        return train_predictions, test_predictions

    def cross_validate(self, model, cv=5):
        """Perform cross-validation."""
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"Cross-Validation Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean()}")
        return scores

    def select_important_features(self, n_features=50):
        """
        Selects top n_features based on importance from a RandomForest model.
        """
        if 'Label' not in self.data.columns:
            raise ValueError("Target column 'Label' is not found in the data.")

        X = self.data.drop(columns=['Label'])  # Excluding the target column
        y = self.data['Label']  # Target column

        model = RandomForestClassifier()
        model.fit(X, y)

        # Getting feature importances
        feature_importances = model.feature_importances_

        # Getting the indices of the top features
        important_features_idx = feature_importances.argsort()[-n_features:][::-1]

        # Getting names of the important features
        important_features = [X.columns[i] for i in important_features_idx]

        # Keeping only the important features along with the target column
        self.data = self.data[important_features + ['Label']]

        print(len(self.data.columns.tolist()))

    # Feature Importance
    def feature_importance(self, model):
        importances = model.feature_importances_
        # indices = np.argsort(importances)[::-1]
        indices = np.argsort(importances)[::-1]
        selected_features = [self.data.columns[i] for i in indices]
        print(len(selected_features))
        # plt.figure()
        # plt.title("Feature importances")
        # plt.bar(range(self.X_test.shape[1]), importances[indices], align="center")
        # plt.xticks(range(self.X_test.shape[1]), [self.X_test.columns[i] for i in indices], rotation=45, ha='right')
        # plt.xlim([-1, self.X_test.shape[1]])
        # plt.show()

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
        print('-------------------------HERE----------------1')
        print((self.data.columns.tolist()))
        return important_features_names
