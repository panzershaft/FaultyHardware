import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from src.models.base_model_interface import BaseModelInterface
from keras import Sequential, Input
from keras.src.layers import Dense


class NeuralNetwork(BaseModelInterface):
    def __init__(self, no_of_input_layer_neurons: int, hidden_layer_activation_func: str,
                 no_of_hidden_layers: int,
                 no_of_hidden_layers_neurons_list: [int],
                 no_of_outer_layer_neurons: int,
                 output_layer_activation_func: str):
        self.no_of_input_layer_neurons = no_of_input_layer_neurons
        self.hidden_layer_activation_func = hidden_layer_activation_func
        self.no_of_hidden_layers = no_of_hidden_layers
        self.no_of_hidden_layers_neurons_list = no_of_hidden_layers_neurons_list
        self.no_of_outer_layer_neurons = no_of_outer_layer_neurons
        self.output_layer_activation_func = output_layer_activation_func
        self.model = Sequential()
        self.setup_model()

    def setup_model(self):
        self.model.add(Input(shape=(self.no_of_input_layer_neurons,), name="Input_layer_1"))
        if (self.no_of_hidden_layers ==
                len(self.no_of_hidden_layers_neurons_list)):
            for num in range(self.no_of_hidden_layers):
                self.model.add(Dense(self.no_of_hidden_layers_neurons_list[num],
                                     activation=self.hidden_layer_activation_func,
                                     name="Hidden_layer_" + str(num + 1)))
            self.model.add(Dense(1, activation=self.output_layer_activation_func, name="Output_layer"))
        else:
            raise ValueError('Number of hidden layers and length of neurons list must match')

    def compile(self, loss: str, optimizer: str, metrics: list, run_eagerly=True):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, batch_size=20,
                       epochs=50, verbose='auto',
                       validation_data=(X_train, y_train),
                       shuffle=True)

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction

    def get_model(self):
        return self.model

    def evaluate(self, X, y):
        predictions = self.predict(X)
        y_pred = np.where(predictions > 0.5, 1, 0)
        eval = self.model.evaluate(X, y, batch_size=20, verbose='auto')
        print(classification_report(y, y_pred))
        print(f'Accuracy: {accuracy_score(y, y_pred):.4f}')
        print(f'F1 Score: {f1_score(y, y_pred):.4f}')
        print("Training AUC: %.2f" % roc_auc_score(y, predictions))
        print(eval)
        # AUC can be calculated if model.predict(X) returns probabilities

    def cross_validate(self, X, y, cv=5):
        pass
        # estimator = KerasClassifier(build_fn=self._create_model, epochs=50, batch_size=32, verbose=0)
        # stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        # results = cross_val_score(estimator, X, y, cv=stratified_kfold)
        # print(f"Cross-Validation Scores: {results}")
        # print(f"Mean Accuracy: {results.mean()}")
