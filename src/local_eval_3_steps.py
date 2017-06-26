from utils import *
import numpy
import random
from random import shuffle
from main import train_and_predict_3_steps

def score_submission(truth, pred):
    score = 0
    for i in xrange(len(truth)):
        if truth[i] == pred[i]:
            score += 0.25
            if truth[i] != 'unrelated':
                score += 0.50
        if truth[i] != 'unrelated' and pred[i] != 'unrelated':
            score += 0.25
    return score

def score_defaults(gold_labels):
    """
    Compute the "all false" baseline (all labels as unrelated) and the max
    possible score
    :param gold_labels: list containing the true labels
    :return: (null_score, best_score)
    """
    unrelated = [g for g in gold_labels if g == 'unrelated']
    null_score = 0.25 * len(unrelated)
    max_score = null_score + (len(gold_labels) - len(unrelated))
    return null_score, max_score

id2body, id2body_sentences = load_body('data/train_bodies.csv')
train_data = load_stance('data/train_stances.csv')[1:];

random.seed(19911226)

fold_scores = []
cm = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
for iteration in xrange(2):
    shuffle(train_data)
    FOLDS = 5
    folds = [[] for i in xrange(FOLDS)]
    for i in xrange(len(train_data)):
        folds[i % FOLDS].append(train_data[i])
    for fold in xrange(FOLDS):
        fold_train, fold_test, fold_truth = [], [], []
        for i in xrange(FOLDS):
            if i != fold:
                fold_train.extend(folds[i])
        for (headline, body_id, stance) in folds[fold]:
            fold_test.append((headline, body_id))
            fold_truth.append(stance)

        fold_pred, _ = train_and_predict_3_steps(fold_train, fold_test, id2body, id2body_sentences)

        test_score = score_submission(fold_truth, fold_pred)

        for i in xrange(len(fold_truth)):
            cm[LABELS.index(fold_truth[i])][LABELS.index(fold_pred[i])] += 1

        null_score, max_score = score_defaults(fold_truth)
        fold_score = test_score / max_score * 100
        print 'iter %d fold %d, score = %.2f%%' % (iteration, fold, fold_score)
        fold_scores.append(fold_score)
        # print SCORE_REPORT.format(max_score, null_score, test_score)

print "mean score: %.2f%%, std-dev: %.2f%%, median: %.2f%%, max: %.2f%%, min: %.2f%%\n" % (numpy.mean(fold_scores), numpy.std(fold_scores), numpy.median(fold_scores), numpy.max(fold_scores), numpy.min(fold_scores))
print_confusion_matrix(cm)
