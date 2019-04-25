import numpy as np
from time import time

from keras.layers import Dense, Input, Concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard


# References:
# Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems."
# Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://keras.io/examples/mnist_denoising_autoencoder/
class DeepCollab:
    def __init__(self, batch_size, hidden_layers, user_features, epochs=100, earlystopping=True, nodes=1024):
        
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.user_features = user_features
        self.epochs = epochs
        self.earlystopping = earlystopping
        self.nodes = nodes

        # Models
        self.autoencoder = None
        

    def fit(self, y, y_noisy, features=None):
        tensorboard = TensorBoard(log_dir="logs/collab/{}".format(time()))
       
        # Input
        input = Input(shape=(y.shape[1],))
        # Encode
        encoded = Dense(y.shape[1], activation='relu')(input)

        if self.user_features:
            user_input = Input(shape=(features.shape[1],))
            user_nodes = Dense(features.shape[1], activation='relu')(user_input)
            merged = Concatenate()([encoded, user_nodes])

            # Hidden Layer
            hidden = Dense(self.nodes, activation='relu')(merged)

        else:
            # Hidden Layer
            hidden = Dense(self.nodes, activation='relu')(encoded)

        if self.hidden_layers == 3:
            hidden = Dense(int(self.nodes/2), activation='relu')(hidden)
            hidden = Dense(self.nodes, activation='relu')(hidden)
        elif self.hidden_layers == 5:
            hidden = Dense(int(self.nodes/2), activation='relu')(hidden)
            hidden = Dense(int(self.nodes/4), activation='relu')(hidden)
            hidden = Dense(int(self.nodes/2), activation='relu')(hidden)
            hidden = Dense(self.nodes, activation='relu')(hidden)

        # Decode
        decoded = Dense(y.shape[1], activation='relu', bias_initializer='ones')(hidden)

        if self.user_features:
            self.autoencoder = Model([input, user_input], decoded)
            
        else:
            self.autoencoder = Model(input, decoded)

        self.autoencoder.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
        self.autoencoder.summary()

        if self.earlystopping:
            callbacks = [EarlyStopping(), tensorboard]
        else:
            callbacks = [tensorboard]
            
        if self.user_features:

            self.autoencoder.fit([y_noisy, features], y,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_split=0.2,
                                 callbacks=callbacks)

        else:

            self.autoencoder.fit(y_noisy, y,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_split=0.2,
                                 callbacks=callbacks)

    def predict(self, y_noisy, features):

        if self.user_features:
            predictions = self.autoencoder.predict([y_noisy, features], verbose=1)
        else:
            predictions = self.autoencoder.predict(y_noisy, verbose=1)

        return predictions
