import numpy as np
from time import time

from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.linear_model import Lasso, LogisticRegression

class WideDeep:
    def __init__(self, num_classes, batch_size):
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.deep = None
        self.linear = None
        self.logistic = None

    def fit(self, train, y):
        tensorboard = TensorBoard(log_dir="logs/0/{}".format(time()))

        # Train a feed forward network with two hidden layers
        # Supposed to use Embeddings for input, but not appropriate for our data set
        self.deep = Sequential()
        self.deep.add(Dense(512, activation='relu', input_shape=(train.shape[1],)))
        self.deep.add(Dropout(0.2))
        self.deep.add(Dense(256, activation='relu'))
        self.deep.add(Dropout(0.2))
        self.deep.add(Dense(128, activation='relu'))
        self.deep.add(Dropout(0.2))
        self.deep.add(Dense(self.num_classes, activation='softmax'))

        self.deep.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        self.deep.fit(train, y,
                      batch_size=self.batch_size,
                      epochs=100,
                      verbose=1,
                      validation_split=0.2,
                      callbacks=[EarlyStopping(), tensorboard])

        # Train a linear regression with l1 regularization
        print('Training linear regression with l1 regularization')
        self.linear = Lasso()
        self.linear.fit(train, y)

        # Supposed to do joint training of logistic regression, but requires tensorflow
        # Instead doing an ensemble method
        print('Training logistic regression')
        deep_predictions = np.array(self.deep.predict(train))
        wide_predictions = self.linear.predict(train)
        # joint_predictions = deep_predictions + wide_predictions  # TODO supposed to be weighted sum?
        joint_predictions = np.concatenate((deep_predictions, wide_predictions), axis=1)

        self.logistic = LogisticRegression()
        self.logistic.fit(joint_predictions, y)

    def predict(self, test):
        deep_predictions = np.array(self.deep.predict(test))
        wide_predictions = self.linear.predict(test)
        # joint_predictions = deep_predictions + wide_predictions
        joint_predictions = np.concatenate((deep_predictions, wide_predictions), axis=1)

        return self.logistic.predict_proba(joint_predictions)

