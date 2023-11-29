from abc import ABC, abstractmethod


class BaseModelInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass

    @abstractmethod
    def cross_validate(self, X, y, cv):
        pass
