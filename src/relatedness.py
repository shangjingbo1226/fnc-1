import xgboost as xgb

def train_relatedness_classifier(trainX, trainY):
    xg_train = xgb.DMatrix(trainX, label=trainY)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 20

    num_round = 1000
    relatedness_classifier = xgb.train(param, xg_train, num_round);

    return relatedness_classifier
