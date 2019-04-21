import numpy as np
from time import time

from keras.layers import Dense, Dropout, Input, Concatenate, Add
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers
import matplotlib.pyplot as plt


# References
# Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems."
# Proceedings of the 1st workshop on deep learning for recommender systems. ACM, 2016.
class WideDeep:
    def __init__(self, num_classes, batch_size, lambda_1, extra_hidden, epochs=100):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lambda_1 = lambda_1
        self.extra_hidden = extra_hidden
        self.epochs = epochs

        self.deep = None

    def fit(self, train, y, early_stopping=True):
        tensorboard = TensorBoard(log_dir="logs/wd/{}".format(time()))

        # Train a feed forward network with two hidden layers
        # Supposed to use Embeddings for input, but not appropriate for our data set
        input_layer = Input(shape=(train.shape[1],))
        deep = Dense(512, activation='relu')(input_layer)
        deep = Dropout(0.2)(deep)

        if self.extra_hidden:
            deep = Dense(256, activation='relu')(deep)
            deep = Dropout(0.2)(deep)

        deep = Dense(128, activation='relu')(deep)
        deep = Dropout(0.2)(deep)
        deep = Dense(self.num_classes, activation='relu')(deep)

        # Train a linear regression with l1 regularization
        # https://stats.stackexchange.com/questions/263211/trying-to-emulate-linear-regression-using-keras
        linear = Dense(1, activation='linear', activity_regularizer=regularizers.l1(self.lambda_1))(input_layer)

        # Train a logistic regression
        if self.num_classes == 1:
            merge = Add()([deep, linear])
            logistic = Dense(self.num_classes, activation='sigmoid')(merge)
            loss = 'binary_crossentropy'
        else:
            merge = Concatenate()([deep, linear])
            logistic = Dense(self.num_classes, activation='softmax')(merge)
            loss = 'categorical_crossentropy'

        # Compile and train the model
        self.deep = Model(inputs=input_layer, outputs=logistic)

        self.deep.compile(loss=loss, optimizer='adagrad', metrics=['accuracy'])

        print(self.deep.summary())

        if early_stopping:
            callbacks = [EarlyStopping(), tensorboard]
        else:
            callbacks = [tensorboard]

        history = self.deep.fit(train, y,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=1,
                                validation_split=0.2,
                                callbacks=callbacks)

        # Reference: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()



    def predict(self, test):

        return self.deep.predict(test, verbose=1)

