import surprise
from surprise.dataset import Dataset
from surprise import SVD
import pandas as pd
import numpy as np
import os

from recsys.evaluate import top_n_5star_results, replay_5star_results


class CollaborativeFiltering:

    def __init__(self):
        """
        Algo: SVD
            n_factors: The number of factors : 100
            n_epochs: The number of iteration of the SGD procedure : 20
            init_mean: The mean of the normal distribution for factor vectors initialization : 0
            init_std_dev: The standard deviation of the normal distribution for factor vectors initialization : 0.1
            lr_all: The learning rate for all parameters : 0.005
            reg_all: The regularization term for all parameters : 0.02
        """
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
        self.algo = SVD()
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
        """
        predict rating of a given book for given user.
        :param user_id: int : id of a user
        :param item_id: int : id of a book
        :return: result : float : predicted rating range(0, 5]
        """
        true_rating = 0
        input_list = [(user_id, item_id, true_rating)]
        result = self.algo.test(input_list)
        return result


if __name__ == "__main__":
    cf = CollaborativeFiltering()


