import numpy as np

from base_model import BaseModel


# References:
# https://github.com/huazhengwang/BanditLib/blob/master/lib/hLinUCB.py
# Huazheng Wang, Qingyun Wu and Hongning Wang. Learning Hidden Features for Contextual Bandits.
# The 25th ACM International Conference on Information and Knowledge Management (CIKM 2016), p1633-1642, 2016.
class HLinUCB(BaseModel):
    def __init__(self, alpha_a, alpha_u, lam1, lam2, context_columns, user_column, decisions_column):
        self.alpha_a = alpha_a
        self.alpha_u = alpha_u
        self.lam1 = lam1
        self.lam2 = lam2
        self.model = {}
        self.arms = []

        # Latent features for each user
        self.user_coef = {}
        self.user_list = np.array([])

        # input processing metadata
        self.context_columns = context_columns
        self.user_column = user_column
        self.decisions_column = decisions_column

    def fit(self, train, y):
        # Separate the inputs
        decisions = np.asarray(train[self.decisions_column])
        users = np.asarray(train[self.user_column])

        self.arms = list(set(decisions))
        self.user_list = np.array((set(users)))

        x = np.array(train[self.context_columns])
        y = np.asarray(y)

        num_features = x.shape[1]

        for arm in self.arms:

            # Initialize ridge regression
            self.model[arm] = HRidge(self.alpha_a, self.alpha_u, self.lam1, self.lam2, num_features)

            # Train on history
            idx = np.where(decisions == arm)
            self.user_coef = self.model[arm].fit(x[idx], y[idx], self.user_list[idx], self.user_coef, self.lam2)

    def predict(self, test, n):
        # Separate the inputs
        x = test[self.context_columns]
        user = test[self.user_column]
        predictions = {}

        # Calculate predicted value for each available arm
        for arm in self.arms:
            predictions[arm], self.user_coef = self.model[arm].predict(x, user, self.user_coef)

        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        top_n = []
        for _ in range(n):
            prediction = max(predictions, key=predictions.get)
            top_n.append(prediction)
            predictions.pop(prediction)

        return top_n


class HRidge:
    def __init__(self, alpha_a, alpha_u, lam1, lam2, num_features):
        self.alpha_a = alpha_a
        self.alpha_u = alpha_u
        self.lam2 = lam2
        self.num_features = num_features

        # Initialize coefficients
        self.c = lam1 * np.identity(self.num_features)
        self.d = np.zeros(self.num_features)
        self.v = np.zeros(self.num_features)

    def fit(self, x, y, users, user_coef):
        # If arm has training data
        if len(x) > 1:

            user_list = list(set(users))

            for user in user_list:

                # Initialize model of latent features for each user
                if user not in user_coef.keys():
                    user_coef[user] = UserLatent(self.alpha_u, self.lam2, self.num_features)

                # Fit the user models
                user_idx = np.where(users == user)
                user_coef[user].fit(x[user_idx], y[user_idx], self.v)

                # Update c, d, v
                self.c = self.c + np.dot(user_coef[user].theta, user_coef[user].T)
                self.d = self.d + np.dot(user_coef[user].theta, (y[user_idx] - np.dot(x.T, user_coef[user].theta)))
                self.v = np.dot(np.linalg.inv(self.c), self.d)

        # Return the updated dictionary of user models
        return user_coef

    def predict(self, x, user, user_coef):

        # Add cold start users
        if user not in user_coef.keys():
            user_coef[user] = UserLatent(self.alpha_u, self.lam2, self.num_features)

        # Calculate the ridge regression predicted value, latent features modifier, and exploration modifier
        ridge = np.dot(np.inner(x, self.v).T, user_coef[user].theta)
        latent = self.alpha_u * np.sqrt(np.dot(np.dot(np.inner(x, self.v),
                                                      np.linalg.inv(user_coef[user].a)), np.inner(x, self.v).T))
        explore = self.alpha_a * np.sqrt(np.dot(np.dot(user_coef[user].theta,
                                                       np.linalg.inv(self.c)), user_coef[user].theta.T))
        prediction = ridge + latent + explore

        # Return predicted value and updated dictionary of user models
        return prediction, user_coef


class UserLatent:
    def __init__(self, alpha, lam, num_features):
        self.alpha = alpha

        # Initialize coefficents
        self.a = lam * np.identity(num_features)
        self.b = np.zeros(num_features)
        self.theta = np.zeros(num_features)

    def fit(self, x, y, v):

        # Update a, b, theta for the user
        self.a = self.a + np.dot(np.inner(x, v), np.inner(x, v).T)
        self.b = self.b + np.dot(np.inner(x, v), y)
        self.theta = np.dot(np.linalg.inv(self.a), self.b)
