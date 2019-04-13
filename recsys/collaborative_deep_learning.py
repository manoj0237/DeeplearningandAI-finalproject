import numpy as np

from keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from base_model import BaseModel



class DeepCollab(BaseModel):
    def __init__(self):
        self.model = None

    def fit(self, train):
        pass

    def predict(self):
        pass

