from sklearn.metrics import average_precision_score, confusion_matrix, recall_score, precision_score
import numpy as np
import pickle

from collaborative_deep_learning import *
from linucb import *
from wide_deep import *


pretrained = False

models = [('hLinUCB', HLinUCB()), ('DeepCollab', DeepCollab()), ('WideDeep', WideDeep())]
# ('Collab', Collab())
# ('DeepQ', DeepQ())

########################
# Data Preprocessing
########################
train = []                  # TODO
test = []                   # TODO
test_actual = []            # TODO

average_precision = {}
replay_test = {}
scoring_test = {}

########################
# Train the models
########################

for name, model in models:
    if pretrained:
        model.load(name + '.h5')
    else:
        model.fit(train)  # TODO this is different for each model
    average_precision[name] = []
    replay_test[name] = []
    scoring_test[name] = []

########################
# Evaluate the models
########################
# TODO This needs to do both replay and top n

def evaluate_user_predictions(ratings, user_id, predictions):
    actual = []
    pred = []
    for p in predictions:
        rating = ratings.loc[ratings['user_id'] == user_id & ratings['book_id'] == p]['rating']
        if len(rating) > 0:
            actual.append(rating[0])
            pred.append(1)

    return actual, pred

for index, row in enumerate(test):
    for name, model in models:
        predictions = model.predict(row)
        scoring, replay = evaluate_user_predictions(test[index], predictions)
        ap = average_precision_score(scoring, replay)
        average_precision[name].append(ap)
        scoring_test[name].append(scoring)
        replay_test[name].append(replay)

mAP = {}
total_recall = {}
total_precision = {}

for name in average_precision.keys():
    print(name, 'Evaluation')
    print('confusion matrix', confusion_matrix(scoring_test[name], replay_test[name]))

    mAP[name] = np.asarray(average_precision[name]).mean()
    print('mAP', mAP[name])

    total_recall[name] = recall_score(scoring_test[name], replay_test[name])
    print('recall', total_recall[name])

    total_precision[name] = precision_score(scoring_test[name], replay_test[name])
    print('precision', total_precision[name])

pickle.dump(mAP, open('mAP.pkl', 'w'))
pickle.dump(total_recall, open('recall.pkl', 'w'))

