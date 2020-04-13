"""
    columns 1-13 - 13 qualities on first image;
    columns 14-26 - 13 qualities on second image;
    columns 27-37 - 11 matching scores between the two images.

    There are in total 9,800,713 training observations.

    Here, the performance criterion is TPR for the value of FPR = 10−4,
    or, speaking in other words, one needs to maximize the value of the
    receiver operating characteristic (ROC) in the point FPR = 10−4.
    The submitted solution file should thus contain the score for each observation.

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import random
import pickle
from xgboost import XGBClassifier
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

start_time = time.time()

def score_for_threshold(threshold, yvalid_sorted):
    N = int(np.sum(yvalid_sorted == 0))
    P = int(np.sum(yvalid_sorted == 1))
    FP = 0
    TP = 0
    for i in range(len(yvalid_sorted) - 1, -1, -1):
        if (yvalid_sorted[i] == 1):
            TP = TP + 1
        else:
            FP = FP + 1
        if (float(FP / N) > 10**-4):
            FP = FP - 1
            break
    return TP / P

# Inspired from https://docs.eyesopen.com/toolkits/cookbook/python/plotting/roc.html
def get_rates(yvalid_sorted):
    tpr=[0.0]
    fpr=[0.0]
    nb_simil=np.sum(yvalid_sorted)
    nb_not_simil=len(yvalid_sorted)-nb_simil
    foundSimil=0
    foundNotSimil=0
    for idx in range(len(yvalid_sorted)-1,-1,-1):
        if(yvalid_sorted[idx]==1):
            foundSimil+=1
        else:
            foundNotSimil+=1
        # in computing the rates every iteration, it's like we set the
        # threshold just above the current value of foundSimil
        tpr.append(foundSimil/float(nb_simil))
        fpr.append(foundNotSimil/float(nb_not_simil))
    return tpr, fpr

def plotRoc(fpr,tpr):
    plt.figure(figsize=(4, 4), dpi=80)
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    # MANUAL PLOT: tpr, fpr = get_rates(yvalid_sorted_with_predict) # MANUAL
    plt.plot(fpr, tpr, linewidth=2)
    plt.axvline(x=threshold, c='r')

def loadModel(modelName):
    return pickle.load(open(modelName, 'rb'))

def dumpModel(model, modelName):
    pickle.dump(model, open(modelName, 'wb'))

def loadTestData(xtest_path):
    x = pd.read_csv(xtest_path)
    return x

def loadTrainingData(UseRandomLoad, UseRandomAfterLoad, xtrain_path, ytrain_path, nb_rows_load,
             nb_rows_after_load, random_state, percent_valid):

    nb_rows_train=int(nb_rows_load*percent_valid)

    if UseRandomLoad:
        random.seed(0)
        n = 9800713 #sum(1 for line in open(XTRAIN_PATH)) - 1
        skip = sorted(random.sample(range(1,n+1),nb_rows_load))
        x = pd.read_csv(XTRAIN_PATH, skiprows=skip)
        y = pd.read_csv(YTRAIN_PATH, skiprows=skip)
    else:
        x = pd.read_csv(XTRAIN_PATH, delimiter=',', nrows=nb_rows_load, skiprows=2000000)
        y = pd.read_csv(YTRAIN_PATH, delimiter=',', nrows=nb_rows_load, skiprows=2000000)

    if UseRandomAfterLoad:
        x = x.sample(nb_rows_after_load, random_state=random_state)
        y = y.sample(nb_rows_after_load, random_state=random_state)
        nb_rows_train=int(nb_rows_after_load/2)

    xtrain=x[:nb_rows_train]
    ytrain=y[:nb_rows_train]

    xvalid=x[nb_rows_train:]
    yvalid=y[nb_rows_train:]

    return xtrain, ytrain, xvalid, yvalid

'********************************************************************'

UseRandomLoad=False
UseRandomAfterLoad=False
XTRAIN_PATH='../../../data_challenge/xtrain_challenge.csv'
XTEST_PATH='../../../data_challenge/xtest_challenge.csv'
YTRAIN_PATH='../../../data_challenge/ytrain_challenge.csv'
nb_rows_load=2000000
nb_rows_after_load=100000
threshold=10**-4
percent_valid=1 # 1 => no validation

random_state=0
random_state_list=list(range(0,1))

'*********************************************************************'

score_list=[]

for random_state in random_state_list:

    # Load
#    xtrain, ytrain, xvalid, yvalid = loadTrainingData(UseRandomLoad=UseRandomLoad, UseRandomAfterLoad=UseRandomAfterLoad, xtrain_path=XTRAIN_PATH,
#                                              ytrain_path=YTRAIN_PATH, nb_rows_load=nb_rows_load, nb_rows_after_load=nb_rows_after_load,
#                                              random_state=random_state, percent_valid=percent_valid)

    xtest=loadTestData(XTEST_PATH)
    # Train
    #clf = LogisticRegression(solver='newton-cg')
    print("finished loading")
    param_test = {
            'max_depth':[6,7,8],
            'learning_rate':[0.1,0.2,0.3]
            'n_estimators':[500,1000,1500],
            'tree_method':'gpu_hist',
            #'n_jobs':-1
    }
    #clf = GridSearchCV(estimator = XGBClassifier(tree_method='gpu_hist'), param_grid = param_test, scoring='roc_auc', cv=3)
    #clf.fit(xtrain, np.array(ytrain).ravel())
    #clf.grid_scores_, clf.best_params_, clf.best_score_
    #clf = XGBClassifier(**param_test)
    clf = loadModel('xgb-4M-gpu.sav')
    #clf.fit(np.array(xtrain), np.array(ytrain).ravel())#clf.fit(xtrain, np.array(ytrain).ravel())

    #dumpModel(clf, 'xgb-4M-gpu.sav')

    # Prediction
    ypred_proba = clf.predict_proba(np.array(xtest))[:,clf.classes_ == 1][:,0] # proba to be 1
    print("finished prediction")

    np.savetxt('res-4M-xgboost-gpu.csv', ypred_proba, fmt = '%1.15f', delimiter=',')

    """
    # Performance
    ypred_proba_sorted = np.sort(ypred_proba)
    yvalid_sorted_with_predict = np.array(np.array(yvalid))[np.argsort(ypred_proba)][:,0]

    fpr, tpr, thresholds = roc_curve(yvalid, ypred_proba, pos_label=1)
    plotRoc(fpr,tpr)
    score = score_for_threshold(threshold,yvalid_sorted_with_predict)
    print("score for random_state {} (threshold {}) is {}".format(random_state,threshold,score))

    score_list.append(score)
    """

print("{} seconds".format(round(time.time() - start_time,2)))