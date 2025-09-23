import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.metrics import recall_score as recall
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer


# important functions
def cv(m, p, xtrain, ytrain, xVald, yVald):
    inner_cv = StratifiedKFold(n_splits=3)
    clf = GridSearchCV(m, 
                       p, 
                       scoring=make_scorer(recall), 
                       n_jobs=-1, 
                       cv=inner_cv, 
                       refit=True, 
                       verbose=0)
    
    clf.fit(X=np.array(xtrain), y=np.array(ytrain).reshape(len(ytrain)).ravel())
    pred = clf.predict(xVald)
    
    return recall(yVald, pred), clf


def get_recall_ml(m, p, xTrain, yTrain, xVald, yVald, xTest, yTest):
    
    recallTot_vald, clfTot_vald = cv(m, p, xTrain, yTrain, xVald, yVald)
    y_Tot_pred = clfTot_vald.predict(xTest)
    df = pd.DataFrame({'SFGs predicted': y_Tot_pred})
    df.to_csv('normalised/yPred_SFGs.csv', index=False, header=True)  # new adjustment
    recallTot_test = recall(yTest, y_Tot_pred)
    
    jackTrainArr = []
    jackValdArr = []
    jackTestArr = []
            
    for i in range(0, len(xTrain), 10):
        x_train = np.delete(np.array(xTrain), i, 0)
        y_train = np.delete(np.array(yTrain), i, 0)
        
        scoreTrain, clrecall = cv(m, p, x_train, y_train, xVald, yVald)
        
        jackTrainArr.append(scoreTrain)

    for t in range(len(xVald)):
        x_vald = np.delete(np.array(xVald), t, 0)
        y_vald = np.delete(np.array(yVald), t, 0)
            
        y_predict = clrecall.predict(x_vald)
        vscore = recall(y_vald, y_predict)
        
        jackValdArr.append(vscore)
        
    for p in range(len(xTest)):
        x_test = np.delete(np.array(xTest), p, 0)
        y_test = np.delete(np.array(yTest), p, 0)
            
        y_pred = clrecall.predict(x_test)
        tscore = recall(y_test, y_pred)
        
        jackTestArr.append(tscore)  
            
    return recallTot_vald, recallTot_test, jackTrainArr, jackValdArr, jackTestArr


def jack_SD (baseTH, ml2):
    baseTH = np.array(baseTH).astype(np.float64)
    ml2 = np.array(ml2).astype(np.float64)

    n = len(baseTH)
    
    deff = ml2-baseTH # element wise, #
#     print(deff)
    mean_jack_stat = np.mean(deff, axis=0)

    
    
    std_err = np.sqrt((n-1)*np.mean( (deff - mean_jack_stat)*(deff - mean_jack_stat), axis=0))
#     print(std_err)
    return [std_err, mean_jack_stat, deff] 


def result_per_split_recall(ml_dicts, models, s):
    
    arr_all = []
    
    for m, d in zip(models, ml_dicts.keys()):
        recall_arr_vald = []
        recall_arr_test = []
        sd_vald_arr = []
        sd_arr = [] 
        key_arr = []
        
        for key in ml_dicts[d].keys():
            str_key = str(key)
            if str_key[0:3] == str(s):
                recall_arr_vald.append(ml_dicts[d][key]['tot_recall_vald'])  # append total validation recall score to an array
                recall_arr_test.append(ml_dicts[d][key]['tot_recall_test'])  # append total test recall score to an array

                sd_train = jack_SD(np.zeros(len(ml_dicts[d][key]['jack_train'])), ml_dicts[d][key]['jack_train'])[0]
                sd_vald = jack_SD(np.zeros(len(ml_dicts[d][key]['jack_vald'])), ml_dicts[d][key]['jack_vald'])[0]
                sd_test = jack_SD(np.zeros(len(ml_dicts[d][key]['jack_test'])), ml_dicts[d][key]['jack_test'])[0]

                sd_v = np.sqrt(np.array((sd_train**2)) + np.array((sd_vald**2)))
                sd = np.sqrt(np.array((sd_train**2)) + np.array((sd_test**2)))

                sd_vald_arr.append(sd_v)
                sd_arr.append(sd)
                key_arr.append(key)

        arr_all.append([key_arr, recall_arr_vald, recall_arr_test, sd_vald_arr, sd_arr])    

    return arr_all


def base_per_split(ml_dicts, models, s):
    # List containing data for different classifiers
    recall_diff_all = []

    # Loop through models to compile all data
    for m, m_key in zip(models, ml_dicts.keys()):
        recall_diff_arr = [] 
        sd_diff_arr = []
        key_arr = []
        
        for f_key in ml_dicts[m_key].keys():
            str_key = str(f_key)
            if str_key[0:3] == str(s):
                base_key = str(s) + ', recall'
                recall_diff_arr.append(ml_dicts[m_key][f_key]['tot_recall_test'] - ml_dicts['lr'][base_key]['tot_recall_test'])
                sd_train = jack_SD(ml_dicts['lr'][base_key]['jack_train'], ml_dicts[m_key][f_key]['jack_train'])[0]
                sd_test = jack_SD(ml_dicts['lr'][base_key]['jack_test'], ml_dicts[m_key][f_key]['jack_test'])[0]

                sd = np.sqrt(np.array((sd_train**2)) + np.array((sd_test**2)))

                sd_diff_arr.append(sd)  # append sd_arr to an array
                key_arr.append(f_key)
                
        recall_diff_all.append([key_arr, recall_diff_arr, sd_diff_arr]) 
    return recall_diff_all


def base_per_split_vald(ml_dicts, models, s):
    # List containing data for different classifiers
    recall_diff_all = []

    # Loop through models to compile all data
    for m, m_key in zip(models, ml_dicts.keys()):
        recall_diff_arr = [] 
        sd_diff_arr = []
        key_arr = []
        
        for f_key in ml_dicts[m_key].keys():
            str_key = str(f_key)
            if str_key[0:3] == str(s):
                base_key = str(s) + ', recall'
                recall_diff_arr.append(ml_dicts[m_key][f_key]['tot_recall_vald'] - ml_dicts['lr'][base_key]['tot_recall_vald'])
                sd_train = jack_SD(ml_dicts['lr'][base_key]['jack_train'], ml_dicts[m_key][f_key]['jack_train'])[0]
                sd_test = jack_SD(ml_dicts['lr'][base_key]['jack_vald'], ml_dicts[m_key][f_key]['jack_vald'])[0]

                sd = np.sqrt(np.array((sd_train**2)) + np.array((sd_test**2)))

                sd_diff_arr.append(sd)  # append sd_arr to an array
                key_arr.append(f_key)
                
        recall_diff_all.append([key_arr, recall_diff_arr, sd_diff_arr]) 
    return recall_diff_all