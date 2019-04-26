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

        self.user_features = None
        self.train_ratings = None
        self.test_ratings = None
        self.cold_ratings = None
        self.train_pivot = None
        self.full_ratings = None
        self.content_features = None
        self.books = None
        self.user_book_info = None

    def run(self):

        self.train_pivot = self.train_ratings.pivot(index='user_id', columns='book_id')
        self.full_ratings = pd.read_csv('data/unprocessed/ratings.csv')
        self.user_features = pd.read_csv('data/user_features_final.csv', header=0)
        self.content_features = pd.read_csv('books_with_latent_features.csv', header=0)
        self.books = pd.read_csv('data/unprocessed/books.csv')

        if self.user_id > 5:
            self.train_ratings = pd.read_csv('data/train_ratings_set.csv')
            self.test_ratings = pd.read_csv('data/test_ratings_set.csv')

            self.user_features = self.user_features[self.train_ratings['user_id'] == self.user_id]
            print('User Features')
            print(self.user_features)
            print('\n')
            self.user_features['average_rating'] /= 5

            self.train_ratings = self.train_ratings[self.train_ratings['user_id'] == self.user_id]
            books = self.train_ratings.merge(self.books, on='book_id')
            print('Historic Rating Data')
            print(books[['title', 'authors', 'rating']])
            print('\n')

            user_and_content = self.user_features.join(self.content_features.drop('book_id', axis=1), how='right')
            user_ratings = self.train_pivot.loc[self.user_id]

            all_books = self.test_ratings[self.test_ratings['user_id'] == self.user_id]['book_id'].tolist()
            idx = [b - 1 for b in all_books]
            filter_books = self.test_ratings[self.test_ratings['user_id'] == self.user_id]
            test_book_ratings = filter_books[filter_books['book_id'].isin(all_books)]['rating'].tolist()
            self.user_book_info = self.books[self.books['book_id'].isin(all_books)][['book_id', 'title', 'authors']]

            self.predict_cf(user_ratings, idx, all_books, test_book_ratings, self.user_id)
            self.predict_lr(user_and_content, idx, all_books, test_book_ratings)
            self.predict_linucb(idx, all_books, test_book_ratings)
            self.predict_cdae(user_ratings, idx, all_books, test_book_ratings)
            self.predict_wd(user_and_content, idx, all_books, test_book_ratings)

        else:
            self.cold_ratings = pd.read_csv('data/cold_start_ratings_set.csv')

            print('Cold Start User Selected')

            all_books = self.cold_ratings[self.cold_ratings['user_id'] == self.user_id]['book_id'].tolist()
            idx = [b - 1 for b in all_books]
            filter_books = self.cold_ratings[self.cold_ratings['user_id'] == self.user_id]
            test_book_ratings = filter_books[filter_books['book_id'].isin(all_books)]['rating'].tolist()
            self.user_book_info = self.books[self.books['book_id'].isin(all_books)][['book_id', 'title', 'authors']]

            self.user_features = self.user_features.drop('user_id', axis=1).mean()
            self.user_features['average_rating'] /= 5

            user_and_content = self.user_features.join(self.content_features.drop('book_id', axis=1), how='right')
            user_ratings = np.zeros((1, 10000))

            print('Collaborative filtering cannot predict cold start users. Proceeding to logistic regression.')

            self.predict_lr(user_and_content, idx, all_books, test_book_ratings)
            self.predict_linucb(idx, all_books, test_book_ratings)
            self.predict_cdae(user_ratings, idx, all_books, test_book_ratings)
            self.predict_wd(user_and_content, idx, all_books, test_book_ratings)

    def predict_cf(self, user_ratings, idx, all_books, test_book_ratings, user_id):
        input_data = [(user_id, book_id, rating) for (book_id, rating) in zip(all_books, test_book_ratings)]
        cf_predictions = self.cf.test(input_data)
        pred_list = [[uid, iid, true_r, est] for uid, iid, true_r, est, _ in cf_predictions]
        df = pd.DataFrame(pred_list, columns=['user_id', 'book_id', 'rating', 'pred'])
        df['pred_proba'] = df['pred'].apply(lambda x: x / 5)
        df['pred'] = df['pred'].apply(lambda x: round(x))
        df['binary_rating'] = df.apply(lambda x: 1 if x['rating'] > 3 else 0, axis=1)
        df = df.sort_values(by='pred_proba')
        df.reset_index(drop=True, inplace=True)
        top_recs = min(len(df), 10)
        df = df[:top_recs]
        df = df.merge(self.user_book_info, on='book_id')
        df = df[['user_id', 'book_id', 'title', 'authors', 'rating', 'binary_rating', 'pred_proba', 'pred']]

        print('SVD Collaboration Filtering Recommendations')
        print(df)
        print('\n')

    def predict_lr(self, user_and_content, idx, all_books, test_book_ratings):
        lr_predictions = self.logistic.predict(np.asarray(user_and_content.drop(['user_id', 'book_id'], axis=1)))

        raw_predictions = lr_predictions[idx].reshape(1, -1)[0]

        df = pd.DataFrame({'book_id': all_books,
                           'rating': test_book_ratings,
                           'pred_proba': raw_predictions})

        df['user_id'] = self.user_id

        df['binary_rating'] = df.apply(lambda x: 1 if x['rating'] > 3 else 0, axis=1)

        min_p = df['pred_proba'].min()
        max_p = df['pred_proba'].max()

        df['prediction'] = df.apply(lambda x: self.scale_prediction_binary(x['pred_proba'], min_p, max_p), axis=1)
        df = df.sort_values(on='pred_proba')
        df.reset_index(drop=True, inplace=True)
        top_recs = min(len(df), 10)
        df = df[:top_recs]
        df = df.merge(self.user_book_info, on='book_id')
        df = df[['user_id', 'book_id', 'title', 'authors', 'rating', 'binary_rating', 'pred_proba', 'pred']]

        print('Logistic Regression Recommendations')
        print(df)
        print('\n')

    def predict_linucb(self, idx, all_books, test_book_ratings):
        mab_predictions = self.linucb.predict_proba(self.user_features)

        df = pd.DataFrame({'book_id': list(mab_predictions[0].keys()),
                           'pred_proba': list(mab_predictions[0].values())})

        df['book_id'] = df.apply(lambda x: int(x['book_id']), axis=1)
        evaluate = pd.DataFrame({'book_id': all_books, 'rating': test_book_ratings})

        df = evaluate.merge(df, on='book_id')
        df['user_id'] = int(self.user_id)

        df['binary_rating'] = df.apply(lambda x: 1 if x['rating'] > 3 else 0, axis=1)

        min_p = df['pred_proba'].min()
        max_p = df['pred_proba'].max()

        df['prediction'] = df.apply(lambda x: self.scale_prediction_binary(x['pred_proba'], min_p, max_p), axis=1)
        df = df.sort_values(on='pred_proba')
        df.reset_index(drop=True, inplace=True)
        top_recs = min(len(df), 10)
        df = df[:top_recs]
        df = df[['user_id', 'book_id', 'title', 'authors', 'rating', 'binary_rating', 'pred_proba', 'pred']]

        print('LinUCB Recommendations')
        print(df)
        print('\n')

    def predict_cdae(self, user_ratings, idx, all_books, test_book_ratings):
        cdae_predictions = self.cdae.predict([user_ratings,
                                                  np.asarray(self.user_features.drop('user_id', axis=1))])

        raw_predictions = cdae_predictions[:,idx]

        df = pd.DataFrame({'book_id': all_books,
                           'rating': test_book_ratings,
                           'pred_proba': raw_predictions[0]})

        df['user_id'] = self.user_id

        df['binary_rating'] = df.apply(lambda x: 1 if x['rating'] > 3 else 0, axis=1)

        min_p = df['pred_proba'].min()
        max_p = df['pred_proba'].max()

        df['prediction'] = df.apply(lambda x: self.scale_prediction_binary(x['pred_proba'], min_p, max_p), axis=1)
        df = df.sort_values(on='pred_proba')
        df.reset_index(drop=True, inplace=True)
        top_recs = min(len(df), 10)
        df = df[:top_recs]
        df = df[['user_id', 'book_id', 'title', 'authors', 'rating', 'binary_rating', 'pred_proba', 'pred']]

        print('Collaborative Denoising Autoencoder Recommendations')
        print(df)
        print('\n')

    def predict_wd(self, user_and_content, idx, all_books, test_book_ratings):
        wd_predictions = self.widedeep.predict(np.asarray(user_and_content.drop(['user_id', 'book_id'], axis=1)))

        raw_predictions = wd_predictions[idx].reshape(1, -1)[0]

        df = pd.DataFrame({'book_id': all_books,
                           'rating': test_book_ratings,
                           'pred_proba': raw_predictions})

        df['user_id'] = self.user_id

        df['binary_rating'] = df.apply(lambda x: 1 if x['rating'] > 3 else 0, axis=1)

        min_p = df['pred_proba'].min()
        max_p = df['pred_proba'].max()

        df['prediction'] = df.apply(lambda x: self.scale_prediction_binary(x['pred_proba'], min_p, max_p), axis=1)
        df = df.sort_values(on='pred_proba')
        df.reset_index(drop=True, inplace=True)
        top_recs = min(len(df), 10)
        df = df[:top_recs]
        df = df[['user_id', 'book_id', 'title', 'authors', 'rating', 'binary_rating', 'pred_proba', 'pred']]

        print('Wide and Deep Recommendations')
        print(df)
        print('\n')

    @staticmethod
    def scale_prediction_binary(x, min_p, max_p):
            raw = (x - min_p) / (max_p - min_p)
            return np.rint(raw)

