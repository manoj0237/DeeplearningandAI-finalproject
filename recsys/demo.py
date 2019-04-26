from wide_deep import WideDeep
from linucb import LinUCB
from collaborative_deep_learning import DeepCollab
from evaluate import *

import pickle
import pandas as pd
from keras.models import load_model

class Demo:
    def __init__(self, user_id):
        self.user_id = user_id

        self.cf = None
        self.logistic = None
        self.linucb = pickle.load(open('models/linucb_alpha05_binary.pkl', 'rb'))
        self.cdae = load_model('models/cdae_binary.h5')
        self.widedeep = load_model('models/wide_deep_1_01_128_False.h5')

        self.train_ratings = pd.read_csv('data/train_ratings_set.csv')
        self.train_pivot = self.train_ratings.pivot(index='user_id', columns='book_id')
        self.full_ratings = pd.read_csv('data/unprocessed/ratings.csv')
        self.user_features = pd.read_csv('data/user_features_final.csv', header=0)
        self.content_features = pd.read_csv('books_with_latent_features.csv', header=0)
        self.books = pd.read_csv('data/unprocessed/books.csv')

    def run(self):
        if self.user_id > 5:
            user_features = self.user_features[self.train_ratings['user_id'] == self.user_id]
            print('User Features')
            print(user_features)
            print('\n')
            user_features['average_rating'] /= 5

            train_ratings = self.train_ratings[self.train_ratings['user_id'] == self.user_id]
            books = train_ratings.merge(self.books, on='book_id')
            print('Historic Rating Data')
            print(books[['title', 'authors', 'rating']])
            print('\n')

            user_content = user_features.join(self.content_features.drop('book_id', axis=1), how='right')
            user_ratings = self.train_pivot[self.train_pivot['user_id'] == self.user_id]

            cf_predictions = self.cf.predict(user_ratings)
            # TODO

            lr_predictions = self.logistic.predict_proba(np.asarray(user_content.drop(['user_id', 'book_id'], axis=1)))
            # TODO

            linucb_predictions = self.linucb.predict_proba(user_features)


            cdae_predictions = self.cdae.predict([user_ratings,
                                                  np.asarray(user_features.drop('user_id', axis=1))])

            wd_predictions = self.widedeep.predict(np.asarray(user_content.drop(['user_id', 'book_id'], axis=1)))



        else:
            print('Cold Start User Selected')






