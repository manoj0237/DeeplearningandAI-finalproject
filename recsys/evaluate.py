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
    precision = precision_score(evaluation_data['rating'], evaluation_data['prediction'], average='weighted')
    recall = recall_score(evaluation_data['rating'], evaluation_data['prediction'], average='weighted')
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
    counter = 0
    for u in users:
        results = evaluation_data[evaluation_data['user_id'] == u]
        books = results['book_id'].tolist()
        probs = np.asarray(results['pred_proba'])
        top_10 = np.argsort(probs)[:10]
        true = {1: [], 2: [], 3: [], 4: [], 5: []}
        pred = {1: [], 2: [], 3: [], 4: [], 5: []}
        top_binary_books = []
        top_binary_ratings = []
        matches = 0
        for i in top_10:
            rating = results[results['book_id'] == books[i]]['rating'].values
            
            if rating > 0:
                matches += 1
                p = int(results[results['book_id'] == books[i]]['prediction'].values)
                r = int(results[results['book_id'] == books[i]]['rating'].values)
                if p == r:
                    true[r].append(1)
                    pred[r].append(1)
                    for i in range(1, 6):
                        if i != r:
                            true[i].append(0)
                            pred[i].append(0)
                else:
                    true[r].append(1)
                    pred[r].append(0)
                    true[p].append(0)
                    pred[p].append(1)
                    for i in range(1, 6):
                        if i != r and i != p:
                            true[i].append(0)
                            pred[i].append(0)
                
                pb = 1 if p > 3 else 0
                rb = 1 if r > 3 else 0

                top_binary_books.append(pb)
                top_binary_ratings.append(rb)
                
        if matches > 0:
            ap_1 = average_precision_score(true[1], pred[1]) if len(true[1]) > 0 else 0
            ap_2 = average_precision_score(true[2], pred[2]) if len(true[2]) > 0 else 0
            ap_3 = average_precision_score(true[3], pred[3]) if len(true[3]) > 0 else 0
            ap_4 = average_precision_score(true[4], pred[4]) if len(true[4]) > 0 else 0
            ap_5 = average_precision_score(true[5], pred[5]) if len(true[5]) > 0 else 0
            weighted_ap = len(true[1])/matches * ap_1 + len(true[2])/matches * ap_2 + len(true[3])/matches * ap_3 + len(true[4])/matches * ap_4 + len(true[5])/matches * ap_5
            if np.isnan(weighted_ap):
                weighted_ap = 0
            aps.append(weighted_ap)
            
            b_ap = average_precision_score(top_binary_ratings, top_binary_books)
            if np.isnan(b_ap):
                b_ap = 0
            binary_aps.append(b_ap)
            
        else:
            not_scored.append(u)
        if counter % 500 == 0:
            print(str(counter), 'users evaluated')
        counter += 1

    mAP = np.asarray(aps).mean()
    binary_mAP = np.asarray(binary_aps).mean()
    return aps, mAP, binary_mAP, binary_aps, len(not_scored)
