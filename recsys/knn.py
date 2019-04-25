import surprise
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
import pandas as pd
import numpy as np
import os

from recsys.evaluate import top_n_5star_results, replay_5star_results


class CollaborativeFilteringKNN:

    def __init__(self):
        prod = False
        self.ml = pd.read_csv("raw_top10M_ratings.csv",
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
            raw_trainset_list = self.trainset_df.values.tolist()

            raw_trainset_list = [(item[0], item[1], item[2], None) for item in raw_trainset_list]

            self.trainset = self.trainset.construct_trainset(raw_trainset_list)

            print("Training set is loaded")

            self.testset_df = pd.read_csv(data_dir + "/test_ratings_set.csv",
                                       header=0,
                                       dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.float32},
                                       names=("user_id", "book_id", "rating"))
            self.testset = Dataset.load_from_df(self.testset_df[["user_id", "book_id", "rating"]],
                                             surprise.Reader(rating_scale=(1, 5)))

            raw_testset_list = self.testset_df.values.tolist()

            raw_testset_list = [(item[0], item[1], item[2], None) for item in raw_testset_list]
            self.testset = self.testset.construct_testset(raw_testset_list)
            print("Test set is loaded")

        self.algo = KNNBasic()
        self._fit()

    def _fit(self):
        print("fit staretd")
        self.algo.fit(self.trainset)
        print("validation staretd")
        self._validate()

    def _validate(self):
        self.predictions = self.algo.test(self.testset)
        pred_list =[]
        for uid, iid, true_r, est, _ in self.predictions:
            pred_list.append([uid, iid, true_r, est])
        df_for_eval = pd.DataFrame(pred_list, columns=['user_id', 'book_id', 'rating', 'est'])
        df_for_eval['pred_proba'] = df_for_eval['est'].apply(lambda x: x/5)
        df_for_eval['prediction'] = df_for_eval['est'].apply(lambda x: round(x))
        df_for_eval= df_for_eval.drop(columns=['est'])
        print(top_n_5star_results(df_for_eval))
        df_for_eval = df_for_eval.drop(columns=['user_id', 'book_id', 'pred_proba'])
        print(replay_5star_results(df_for_eval))

    def predict_book_rating(self, user_id=80, item_id=34):
        true_rating = 0
        input = [(user_id, item_id, true_rating)]
        result = self.algo.test(input)
        return result[0].est


if __name__ == "__main__":
    cf = CollaborativeFilteringKNN()
