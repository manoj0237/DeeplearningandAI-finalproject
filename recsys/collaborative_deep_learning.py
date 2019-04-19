import numpy as np
from time import time

from keras.layers import Dense, Input, Concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard


# References
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://keras.io/examples/mnist_denoising_autoencoder/
class DeepCollab:
    def __init__(self, batch_size, hidden_layers, step_factor, cdae, user_id_column, user_features, feature_columns=None):
        assert(hidden_layers in [1, 3], 'Invalid number of hidden layers')

        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.step_factor = step_factor

        self.cdae = cdae
        self.user_id_column = user_id_column
        self.user_features = user_features
        self.feature_columns = feature_columns

        # Models
        self.user_models = {}
        self.autoencoder = None
        self.cdae_model = None
        self.cdae_weights = False

    def fit(self, train, y, y_noisy):
        tensorboard = TensorBoard(log_dir="logs/0/{}".format(time()))

        user_id = train[self.user_id_column][0]
        features = train[self.feature_columns]

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
        decoded = Dense(y.shape[1], activation='sigmoid', bias_initializer='ones')(hidden)

        if self.user_features:
            self.autoencoder = Model([input, user_input], decoded)
            if self.cdae:
                self.cdae_model = Model(input, decoded)
                self.user_models[user_id] = Model(user_input, user_nodes)

        else:
            self.autoencoder = Model(input, decoded)

        self.autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')

        if self.cdae_weights:
            self.cdae_model.load_weights('cdae_weights.h5')

        if self.cdae:

            self.autoencoder.fit([y_noisy, features], y,
                                 epochs=5,
                                 batch_size=32,
                                 validation_split=0.2,
                                 callbacks=[EarlyStopping(), tensorboard])

            self.cdae_model.save_weights('cdae_weights.h5')
            self.cdae_weights = True


        elif self.user_features:

            self.autoencoder.fit([y_noisy, features], y,
                                 epochs=100,
                                 batch_size=32,
                                 validation_split=0.2,
                                 callbacks=[EarlyStopping(), tensorboard])

        else:

            self.autoencoder.fit(y_noisy, y,
                     epochs=100,
                     batch_size=32,
                     validation_split=0.2,
                     callbacks=[EarlyStopping(), tensorboard])

    def evaluate(self, test, user_id, y, y_noisy):
        user_id = test[self.user_id_column][0]
        features = test[self.feature_columns]

        if self.cdae:
            pass
        elif self.user_features:
            scores = self.autoencoder.evaluate((y_noisy, features), y, verbose=1)
        else:
            scores = self.autoencoder.evaluate(y_noisy, y, verbose=1)

        print(scores)

    def predict(self, test, user_id, y_noisy, n):
        user_id = test[self.user_id_column][0]
        features = test[self.feature_columns]

        if self.cdae:
            user_model = self.user_models[user_id]
            # TODO how to switch them out?
            predictions = self.autoencoder.predict((y_noisy, features), verbose=1)
        elif self.user_features:
            predictions = self.autoencoder.predict((y_noisy, features), verbose=1)
        else:
            predictions = self.autoencoder.predict(y_noisy, verbose=1)

        new_predictions = {}
        top_n = []

        for index, row in enumerate(predictions):
            if y_noisy[index] == 0:
                new_predictions[index] = row

        for _ in range(n):
            p = max(new_predictions, key=new_predictions.get)
            top_n.append(p)
            new_predictions.pop(p)

        return top_n




