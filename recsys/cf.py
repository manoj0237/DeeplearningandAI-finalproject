import surprise
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy
from surprise import accuracy
import pandas as pd
import numpy as np
from collections import defaultdict
import os

from recsys.evaluate import top_10_5star_results


class CollaborativeFiltering:

    def __init__(self):
        prod = False
        self.ml = pd.read_csv("ratings1k.csv",
                         header=0,
                         dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.float32},
                         names=("user_id", "book_id", "rating"))
        self.data = Dataset.load_from_df(self.ml[["user_id", "book_id", "rating"]],
                                         surprise.Reader(rating_scale=(1, 5)))
        self.cv = 5
        self.trainset, self.testset = train_test_split(self.data, test_size=0.25)

        if prod is True:
            data_dir = os.path.abspath("data/")
            self.trainset_df = pd.read_csv(data_dir + "/train_ratings_set.csv",
                         header=0,
                         dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.float32},
                         names=("user_id", "book_id", "rating"))
            self.trainset = Dataset.load_from_df(self.trainset_df[["user_id", "book_id", "rating"]],
                                         surprise.Reader(rating_scale=(1, 5)))

            self.testset_df = pd.read_csv(data_dir + "/test_ratings_set.csv",
                                       header=0,
                                       dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.float32},
                                       names=("user_id", "book_id", "rating"))
            self.testset = Dataset.load_from_df(self.testset_df[["user_id", "book_id", "rating"]],
                                             surprise.Reader(rating_scale=(1, 5)))

        self.algo = SVD()
        self._fit()

    def _fit(self):
        self.algo.fit(self.trainset)
        self._validate()

    def _validate(self):
        self.predictions = self.algo.test(self.testset)
        pred_list =[]
        for uid, iid, true_r, est, _ in self.predictions:
            pred_list.append([uid, iid, true_r, est])
        df_for_eval = pd.DataFrame(pred_list, columns=['user_id', 'book_id', 'iid', 'est'])
        df_for_eval['pred_proba'] = df_for_eval['est'].apply(lambda x: x/5)
        df_for_eval['prediction'] = df_for_eval['est'].apply(lambda x: round(x))
        df_for_eval['rating'] = df_for_eval['iid'].apply(lambda x: 1 if(x >= 4) else 0)
        df_for_eval= df_for_eval.drop(columns=['iid', 'est'])
        print(df_for_eval)
        print(top_10_5star_results(df_for_eval))


    def predict_book_rating(self, user_id=80, item_id=34):
        true_rating = 0
        input = [(user_id, item_id, true_rating)]
        result = self.algo.test(input)
        print(result[0].est)

    ## TODO: fix this
    def top_recomm(self, user_id=2, n=10):
        top_book_list = []
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n[user_id]
        # return top_book_list



if __name__ == "__main__":
    cf = CollaborativeFiltering()
    # cf.predict_book_rating(user_id=1000000, item_id=34)


    # cf.predict_book_rating()
    # print(cf.top_recomm())



