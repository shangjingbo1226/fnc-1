import os
import nltk
import numpy as np
import xgboost as xgb
from collections import defaultdict
from math import log
from utils import LABELS
from features import extract_features
from relatedness import train_relatedness_classifier
from gensim.models.keyedvectors import KeyedVectors
from utils import normalize_word

word2vec = {}

# Load Google's pre-trained Word2Vec model.
def initialize():
    global word2vec
    if len(word2vec) == 0:
        print 'loading word2vec...'
        word_vectors = KeyedVectors.load_word2vec_format('./resources/GoogleNews-vectors-negative300.bin', binary=True)
        for word in word_vectors.vocab:
            word2vec[normalize_word(word)] = word_vectors[word]
        print 'word2vec loaded'

def prepare_idf(id2body):
    idf = defaultdict(float)
    for (body_id, body) in id2body.iteritems():
        for word in set(body):
            idf[word] += 1
    for word in idf:
        idf[word] = log(len(id2body) / idf[word])
    return idf

import xgboost as xgb

def train_and_predict_3_steps(train, test, id2body, id2body_sentences):
    idf = prepare_idf(id2body)
    global word2vec
    initialize()

    trainX = [extract_features(clean_title, id2body[body_id], id2body_sentences[body_id],idf, word2vec) for (clean_title, body_id, stance) in train]
    trainY = [int(stance == 'unrelated') for (clean_title, body_id, stance) in train]
    relatedness_classifier = train_relatedness_classifier(trainX, trainY)

    relatedTrainX = [trainX[i] for i in xrange(len(trainX)) if trainY[i] == 0]
    relatedTrainY = [int(train[i][2] == 'discuss') for i in xrange(len(trainX)) if trainY[i] == 0]
    discuss_classifier = train_relatedness_classifier(relatedTrainX, relatedTrainY)

    agreeTrainX = [trainX[i] for i in xrange(len(trainX)) if train[i][2] == 'agree' or train[i][2] == 'disagree']
    agreeTrainY = [int(train[i][2] == 'agree') for i in xrange(len(trainX)) if train[i][2] == 'agree' or train[i][2] == 'disagree']
    agree_classifier = train_relatedness_classifier(agreeTrainX, agreeTrainY)

    testX = [extract_features(clean_title, id2body[body_id], id2body_sentences[body_id],idf, word2vec) for (clean_title, body_id) in test]

    xg_test = xgb.DMatrix(testX)
    relatedness_pred = relatedness_classifier.predict(xg_test);
    discuss_pred = discuss_classifier.predict(xg_test)
    agree_pred = agree_classifier.predict(xg_test)

    ret, scores = [], []
    for (pred_relate, pred_discuss, pred_agree) in zip(relatedness_pred, discuss_pred, agree_pred):
        scores.append((pred_relate, pred_discuss, pred_agree))
        if pred_relate >= 0.5:
            ret.append('unrelated')
        elif pred_discuss >= 0.5:
            ret.append('discuss')
        elif pred_agree >= 0.5:
            ret.append('agree')
        else:
            ret.append('disagree')

    return ret, scores
