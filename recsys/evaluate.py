from sklearn.metrics import average_precision_score, confusion_matrix, recall_score, precision_score
import numpy as np


def replay_binary_results(evaluation_data):
    """
    To be used for predictions for the books rated in the test set.

    Parameter
    ---------
    evaluation_data: pd.DataFrame
        Expected columns: prediction, binary_rating
        prediction is the predicted class (0/1)
        binary_rating is the binarized ratings where [1, 2, 3] -> 0, [4, 5] -> 1


    Returns
    -------
    Confusion matrix, precision and recall
    """
    cfm = confusion_matrix(evaluation_data['binary_rating'], evaluation_data['prediction'])
    precision = precision_score(evaluation_data['binary_rating'], evaluation_data['prediction'])
    recall = recall_score(evaluation_data['binary_rating'], evaluation_data['prediction'])
    return cfm, precision, recall


def replay_5star_results(evaluation_data):
    """
    To be used for predictions for the books rated in the test set.

    Parameter
    ---------
    evaluation_data: pd.DataFrame
        Expected columns: prediction, rating
        prediction is the predicted class (1-5)
        rating is original rating

    Returns
    -------
    Confusion matrix, precision and recall
    """
    cfm = confusion_matrix(evaluation_data['rating'], evaluation_data['prediction'])
    precision = precision_score(evaluation_data['rating'], evaluation_data['prediction'])
    recall = recall_score(evaluation_data['rating'], evaluation_data['prediction'])
    return cfm, precision, recall


def top_10_binary_results(evaluation_data):
    """
    Evaluates the top 10 recommendations for each user.
    To be used for predictions for all books not in the user's training data.

    Parameter
    ---------
    evaluation_data: pd.DataFrame
        Expected columns: user_id, book_id, pred_proba, prediction, binary_rating
        pred_proba is the probability of being a 1
        prediction is the predicted class (0/1)
        binary_rating is the binarized ratings where [1, 2, 3] -> 0, [4, 5] -> 1

    Returns
    -------
    The mAP and the number of users who had no historic rating data for any of their predicted top 10
    """
    users = list(set(evaluation_data['user_id'].tolist()))

    aps = []
    not_scored = []
    counter = 0
    for u in users:
        results = evaluation_data[evaluation_data['user_id'] == u]
        books = results['book_id'].tolist()
        probs = np.asarray(results['pred_proba'])
        top_10 = np.argsort(probs)[:10]
        top_books = []
        top_ratings = []
        for i in top_10:
            rating = results[results['book_id'] == books[i]]['binary_rating'].values
            if rating >= 0:
                top_books.append(results[results['book_id'] == books[i]]['prediction'].values)
                top_ratings.append(results[results['book_id'] == books[i]]['binary_rating'].values)
        if len(top_ratings) > 0:
            ap = average_precision_score(np.array(top_ratings), np.array(top_books))
            if np.isnan(ap):
                aps.append(0)
            else:
                aps.append(ap)
        else:
            not_scored.append(u)

        if counter % 500 == 0:
            print(str(counter), 'users evaluated')
        counter +=1

    mAP = np.asarray(aps).mean()
    return mAP, len(not_scored)



def top_10_5star_results(evaluation_data):
    """
    Evaluates the top 10 recommendations for each user.
    To be used for predictions for all books not in the user's training data.

    Parameter
    ---------
    evaluation_data: pd.DataFrame
        Expected columns: user_id, book_id, pred_proba, prediction, binary_rating
        pred_proba is the probability of being a 5
        prediction is the predicted class (1-5)
        rating is the original rating

    Returns
    -------
    The mAP and the number of users who had no historic rating data for any of their predicted top 10.
    Also calculates mAP for whether or not the user enjoyed the book (4 or 5).
    """
    users = list(set(evaluation_data['user_id'].tolist()))

    aps = []
    binary_aps = []
    not_scored = []
    for u in users:
        results = evaluation_data[evaluation_data['user_id'] == u]
        books = results['book_id'].tolist()
        probs = np.asarray(results['pred_proba'])
        top_10 = np.argsort(probs)[:10]
        top_books = []
        top_ratings = []
        top_binary_books = []
        top_binary_ratings = []
        for i in top_10:
            rating = results[results['book_id'] == books[i]]['binary_rating'].values
            if rating >= 0:
                p = results[results['book_id'] == books[i]]['prediction'].values
                r = results[results['book_id'] == books[i]]['binary_rating'].values
                pb = 1 if p > 3 else 0
                rb = 1 if r > 3 else 0
                top_books.append(p)
                top_ratings.append(r)
                top_binary_books.append(pb)
                top_binary_ratings.append(rb)

        if len(top_ratings) > 0:
            aps.append(average_precision_score(top_ratings, top_books))
            binary_aps.append(average_precision_score(top_binary_ratings, top_binary_books))
        else:
            not_scored.append(u)

    mAP = np.asarray(aps).mean()
    binary_mAP = np.asarray(binary_aps).mean()
    return mAP, binary_mAP, len(not_scored)
