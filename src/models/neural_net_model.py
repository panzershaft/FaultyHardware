from src.models.base_model_interface import BaseModelInterface
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.utils import to_categorical


class NeuralNetwork(BaseModelInterface):
    def __init__(self, no_of_input_layer_neurons: int, activation_func: str,
                 no_of_hidden_layers: int,
                 no_of_hidden_layers_neurons_list: [int],
                 no_of_outer_layer_neurons: int):
        self.model = Sequential()
        self.no_of_input_layer_neurons = no_of_input_layer_neurons
        self.activation_func = activation_func
        self.no_of_hidden_layers = no_of_hidden_layers
        self.no_of_hidden_layers_neurons_list = (
            no_of_hidden_layers_neurons_list)
        self.no_of_outer_layer_neurons = no_of_outer_layer_neurons
        self.setup_model()

    def setup_model(self):
        self.model.add(Input(self.no_of_input_layer_neurons))
        if (self.no_of_hidden_layers ==
                len(self.no_of_hidden_layers_neurons_list)):
            for num in range(self.no_of_hidden_layers):
                self.model.dense(self.no_of_hidden_layers_neurons_list[num],
                                 activation=self.activation_func,
                                 name="Hidden_layer_" + str(num))
            self.model.add(Dense(1, activation='sigmoid', name="Output_layer"))
        else:
            print('Ensure that the no_of_hidden_layers is same as len of '
                  'no_of_hidden_layers_neurons_list')

    def train(self, X_train, y_train):
        self.model.model.fit(X_train, y_train, verbose='auto',
                             validation_data=(X_train, y_train),
                             shuffle=True)

    def predict(self, X):
        pass

    def get_model(self):
        pass

    def evaluate(self, X, y):
        self.model.evaluate(X, y, batch_size=20, verbose='auto')

    def cross_validate(self, X, y, cv):
        pass
