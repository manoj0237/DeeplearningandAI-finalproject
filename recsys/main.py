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
        model.fit(train)
    average_precision[name] = []
    replay_test[name] = []
    scoring_test[name] = []

########################
# Evaluate the models
########################


# TODO this is pseudocode, not implementation
def evaluate(test, pred):
    r = []
    s = []
    for prediction in pred:
        if pred in test:
            r.append(prediction)
            s.append(test)
    return s, r


for index, row in enumerate(test):
    for name, model in models:
        predictions = model.predict(row)
        scoring, replay = evaluate(test[index], predictions)
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

