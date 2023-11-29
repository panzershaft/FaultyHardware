from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.base_model_interface import BaseModelInterface
import xgboost as xgb


class XGBoostModel(BaseModelInterface):

    def __init__(self):
        self.model = xgb.XGBClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction

    def get_model(self):
        return self.model

    def evaluate(self, X, y):
        predictions = self.predict(X)

        print(classification_report(y, predictions))
        print(f'Accuracy: {accuracy_score(y, predictions):.4f}')
        print(f'F1 Score: {f1_score(y, predictions):.4f}')
        print("Training AUC: %.2f" % roc_auc_score(y, self.get_model().predict_proba(X)[:, 1]))

    def cross_validate(self, X, y, cv=5):
        stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=stratified_kfold, scoring='accuracy')
        print(f"Cross-Validation Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean()}")
