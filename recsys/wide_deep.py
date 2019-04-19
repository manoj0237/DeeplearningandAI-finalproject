import numpy as np
from time import time

from keras.layers import Dense, Dropout, Input, Concatenate
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers


# References
# Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems."
# Proceedings of the 1st workshop on deep learning for recommender systems. ACM, 2016.
class WideDeep:
    def __init__(self, num_classes, batch_size, lambda_1, extra_hidden):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lambda_1 = lambda_1
        self.extra_hidden = extra_hidden

        self.deep = None
        self.linear = None
        self.logistic = None

    def fit(self, train, y):
        tensorboard = TensorBoard(log_dir="logs/wd/{}".format(time()))

        # Train a feed forward network with two hidden layers
        # Supposed to use Embeddings for input, but not appropriate for our data set
        input = Input(shape=(train.shape[1],))
        deep = Dense(512, activation='relu')(input)
        deep = Dropout(0.2)(deep)

        if self.extra_hidden:
            deep = Dense(256, activation='relu')(deep)
            deep = Dropout(0.2)(deep)

        deep = Dense(128, activation='relu')(deep)
        deep = Dropout(0.2)(deep)
        deep  = Dense(self.num_classes, activation='relu')(deep)

        # Train a linear regression with l1 regularization
        # https://stats.stackexchange.com/questions/263211/trying-to-emulate-linear-regression-using-keras
        linear = Dense(1, activation='linear', activity_regularizer=regularizers.l1(self.lambda_1))(input)

        # Train a logistic regression
        # Paper says to get weighted sum of deep and linear, but this only makes sense for clicks, not ratings
        logistic = Concatenate([deep, linear])
        logistic = Dense(self.num_classes, activation='softmax')(logistic)

        # Compile and train the model
        self.deep = Model(input, logistic)

        self.deep.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

        self.deep.fit(train, y,
                      batch_size=self.batch_size,
                      epochs=100,
                      verbose=1,
                      validation_split=0.2,
                      callbacks=[EarlyStopping(), tensorboard])

    def predict(self, test):

        return self.deep.predict(test, verbose=1)

