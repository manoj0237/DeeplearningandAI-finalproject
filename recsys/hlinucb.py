import numpy as np
import operator
import warnings

warnings.simplefilter("error", RuntimeWarning)


# References:
# Huazheng Wang, Qingyun Wu and Hongning Wang. Learning Hidden Features for Contextual Bandits.
# The 25th ACM International Conference on Information and Knowledge Management (CIKM 2016), p1633-1642, 2016.
# https://github.com/huazhengwang/BanditLib/blob/master/lib/hLinUCB.py
class HLinUCB:
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

        counter = 0
        for arm in self.arms:

            # Initialize ridge regression
            self.model[arm] = HRidge(self.alpha_a, self.alpha_u, self.lam1, self.lam2, num_features)

        for index, row in enumerate(decisions):
            # Train on history

            arm_x = x[index]
            arm_y = y[index]
            arm_users = users[index]

            self.model[row].fit(arm_x, arm_y, arm_users, self.user_coef)

            counter +=1
            if counter % 1000 == 0:
                print(str(counter), 'rows trained')


    def predict(self, test, n):
        # Separate the inputs
        x = np.asarray(test[self.context_columns])
        user = test[self.user_column]
        predictions = []
        for index, row in enumerate(user):
            user_predictions = {}

            # Calculate predicted value for each available arm
            for arm in self.arms:
                user_predictions[arm] = self.model[arm].predict(x[index], row, self.user_coef)

            # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
            top_n = []
            for _ in range(n):
                prediction = max(user_predictions, key=user_predictions.get)
                top_n.append(prediction)
                user_predictions.pop(prediction)

            predictions.append(top_n)

        return predictions

    def predict_proba(self, test):
        x = np.asarray(test[self.context_columns])
        user = test[self.user_column].values
        predictions = []
        for index, row in enumerate(user):
            user_predictions = {}

            # Calculate predicted value for each available arm
            for arm in self.arms:
                user_predictions[arm] = self.model[arm].predict(x[index], row, self.user_coef)
            predictions.append(user_predictions)

        return predictions
    
    def predict_proba_replay(self, test, ratings):
        x = np.asarray(test[self.context_columns])
        user = test[self.user_column].values
        predictions = []
        for index, row in enumerate(user):
            user_predictions = {}

            user_arms = ratings[ratings['user_id'] == row]['book_id'].tolist()

            # Calculate predicted value for each available arm
            for arm in user_arms:
                user_predictions[arm] = self.model[arm].predict(x[index], row, self.user_coef)
            predictions.append(user_predictions)
            if len(predictions) % 500 == 0:
                print(len(predictions), 'users predicted')

        return predictions


class HRidge:
    def __init__(self, alpha_a, alpha_u, lam1, lam2, num_features):
        self.alpha_a = alpha_a
        self.alpha_u = alpha_u
        self.lam2 = lam2
        self.num_features = num_features

        # Initialize coefficients
        self.c = lam1 * np.identity(self.num_features)
        self.d = np.zeros(self.num_features)
        self.v = np.random.normal(0, 1, self.num_features) # np.ones(self.num_features)

    def fit(self, x, y, user, user_coef):
        # If arm has training data
        if len(x) > 1:
            #
            # user_list = list(set(users))
            #
            # for user in user_list:

            # Initialize model of latent features for each user
            if user not in user_coef.keys():
                user_coef[user] = UserLatent(self.alpha_u, self.lam2, self.num_features)

            # Fit the user models

            user_coef[user].fit(x, y, self.v)

            # Update c, d, v
            self.c = self.c + np.dot(user_coef[user].theta, user_coef[user].theta)
            d_update = (user_coef[user].theta * (y - np.dot(x, user_coef[user].theta.reshape(-1, 1)))).T
            self.d = self.d + d_update.T.reshape(1, -1)[0]
            self.v = np.dot(np.linalg.inv(self.c), self.d)


    def predict(self, x, user, user_coef):

        # Add cold start users
        if user not in user_coef.keys():
            user_coef[user] = UserLatent(self.alpha_u, self.lam2, self.num_features)

        # Calculate the ridge regression predicted value, latent features modifier, and exploration modifier
        ridge = np.dot(self.v, user_coef[user].theta)
        latent = 0

        # If no latent features are found, set latent to 0
        try:
            latent = self.alpha_u * np.sqrt(np.dot(np.dot(self.v, np.linalg.inv(user_coef[user].a)), self.v))
        except RuntimeWarning:
            pass

        explore = self.alpha_a * np.sqrt(np.dot(np.dot(x.T, np.linalg.inv(self.c)), x))
        #print(explore)
        prediction = ridge + latent + explore

        # Return predicted value and updated dictionary of user models
        return prediction


class UserLatent:
    def __init__(self, alpha, lam, num_features):
        self.alpha = alpha

        # Initialize coefficents
        self.a = lam * np.identity(num_features)
        self.b = np.zeros(num_features)
        self.theta = np.zeros(num_features)

    def fit(self, x, y, v):

        # Update a, b, theta for the user
        xv = np.dot(x, v)
        self.a = self.a + np.dot(xv, xv)
        self.b = self.b + y * xv
        self.theta = np.dot(np.linalg.inv(self.a), self.b)
