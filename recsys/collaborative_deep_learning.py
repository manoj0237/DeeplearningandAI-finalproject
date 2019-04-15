import numpy as np

from keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM
from keras.models import Model
from keras.models import load_model

from base_model import BaseModel


class DeepCollab(BaseModel):
    def __init__(self):
        self.model = None

    def fit(self, train, y):
        pass

    def predict(self, test):
        pass

