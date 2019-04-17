import numpy as np

from keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM, Input, Concatenate
from keras.models import Model
from keras.models import load_model

from base_model import BaseModel


# References
# https://blog.keras.io/building-autoencoders-in-keras.html
class DeepCollab(BaseModel):
    def __init__(self, batch_size, hidden_layers, step_factor, user_features, feature_columns=None):
        assert(hidden_layers in [1, 3], 'Invalid number of hidden layers')
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.step_factor = step_factor
        self.user_features = user_features
        self.feature_columns = feature_columns

    def fit(self, train, y):

        # Input
        input = Input(shape=(y.shape[1],))
        # Encode
        encoded = Dense(y.shape[1], activation='relu')(input)

        if self.user_features:
            user_input = Input(shape=(len(self.feature_columns),))
            user_nodes = Dense(shape=(len(self.feature_columns),), activation='relu')(user_input)
            merged = Concatenate([encoded, user_nodes])

            # Hidden Layer
            hidden = Dense(int(y.shape[1]/self.step_factor), activation='relu')(merged)

        else:
            # Hidden Layer
            hidden = Dense(int(y.shape[1]/self.step_factor), activation='relu')(encoded)

        if self.hidden_layers ==3:
            hidden = Dense(int(y.shape[1]/(self.step_factor * 2)), activation='relu')(hidden)
            hidden = Dense(int(y.shape[1]/(self.step_factor )), activation='relu')(hidden)

        # Decode
        decoded = Dense(y.shape[1], activation='sigmoid')(hidden)

        if self.user_features:
            autoencoder = Model([input, user_input], decoded)
        else:
            autoencoder = Model(input, decoded)

        autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')
        autoencoder.fit() #TODO

    def predict(self, test, n):
        pass

