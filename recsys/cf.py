import surprise
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy
from surprise import accuracy
import pandas as pd
import numpy as np
from collections import defaultdict


class CollaborativeFiltering:

    def __init__(self):
        self.ml = pd.read_csv("ratings1k.csv",
                         header=0,
                         dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.float32},
                         names=("user_id", "book_id", "rating"))
        self.data = Dataset.load_from_df(self.ml[["user_id", "book_id", "rating"]],
                                         surprise.Reader(rating_scale=(1, 5)))
        self.cv = 5
        self.trainset, self.testset = train_test_split(self.data, test_size=0.25)
        self.algo = SVD()

    def fit(self):
        self.algo.fit(self.trainset)
        self.predictions = self.algo.test(self.testset)
        accuracy.rmse(self.predictions)

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
    cf.fit()
    print(cf.top_recomm())



