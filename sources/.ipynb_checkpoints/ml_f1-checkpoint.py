import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer


# important functions
def cv (m, p, xtrain, ytrain, xVald, yVald):
    inner_cv = StratifiedKFold(n_splits=3)
    clf = GridSearchCV(m, 
                       p, 
                       scoring=make_scorer(f1), 
                       n_jobs=-1, 
                       cv=inner_cv, 
                       refit=True, 
                       verbose=0)
    
    clf.fit(X = np.array(xtrain), y=np.array(ytrain).reshape(len(ytrain)).ravel())
    pred = clf.predict(xVald)
    
    return f1(yVald, pred), clf


def get_f1_ml (m, p, xTrain, yTrain, xVald, yVald, xTest, yTest ):
    
    f1Tot_vald, clfTot_vald = cv(m, p, xTrain, yTrain, xVald, yVald)
    y_Tot_pred = clfTot_vald.predict(xTest)
    f1Tot_test = f1(yTest, y_Tot_pred)
    
    jackTrainArr = []
    jackValdArr = []
    jackTestArr = []
            
    for i in range(0, len(xTrain), 10):
        x_train = np.delete(np.array(xTrain), i, 0)
        y_train = np.delete(np.array(yTrain), i, 0)
        
        scoreTrain, clf1 = cv(m, p, x_train, y_train, xVald, yVald)
        
        jackTrainArr.append(scoreTrain)

    for t in range (len(xVald)):
        x_vald = np.delete(np.array(xVald), t, 0)
        y_vald = np.delete(np.array(yVald), t, 0)
            
        y_predict = clf1.predict(x_vald)
        vscore = f1(y_vald, y_predict)
        
        jackValdArr.append(vscore)
        
    for p in range (len(xTest)):
        x_test = np.delete(np.array(xTest), p, 0)
        y_test = np.delete(np.array(yTest), p, 0)
            
        y_pred = clf1.predict(x_test)
        tscore = f1(y_test, y_pred)
        
        jackTestArr.append(tscore)  
            
    return  f1Tot_vald, f1Tot_test, jackTrainArr, jackValdArr, jackTestArr


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


def result_per_split(ml_dicts,models, s):
    
    arr_all = []
    
    for m, d in zip (models, ml_dicts.keys()):
        f1_arr_vald = []
        f1_arr_test = []
        sd_vald_arr = []
        sd_arr = [] 
        key_arr = []
        # print(ml_dicts[d])
        for key in ml_dicts[d].keys():
            str_key = str(key)
            if str_key[0:3] == str(s):
                f1_arr_vald.append(ml_dicts[d][key][ 'tot_f1_vald' ]) # append total valdation f1 score to an array
                f1_arr_test.append(ml_dicts[d][key][ 'tot_f1_test' ]) # append total test f1 score to an array

                sd_train = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_train' ]) ), ml_dicts[d][key][ 'jack_train' ])[0]
                sd_vald = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_vald' ]) ), ml_dicts[d][key][ 'jack_vald' ])[0]
                sd_test = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_test' ]) ), ml_dicts[d][key][ 'jack_test' ])[0]

                sd_v = np.sqrt( np.array((sd_train**2)) + np.array((sd_vald**2)))
                sd = np.sqrt( np.array((sd_train**2)) + np.array((sd_test**2)))

                sd_vald_arr.append(sd_v)
                sd_arr.append(sd)
                key_arr.append(key)

        arr_all.append([key_arr, f1_arr_vald, f1_arr_test, sd_vald_arr, sd_arr])    

    return arr_all

def base_per_split(ml_dicts, models, s):

    # List containing data for different classifiers
    f1_diff_all = []

    # Loop through models to complie all data
    for m, m_key in zip (models, ml_dicts.keys()):
        f1_diff_arr = [] 
        sd_diff_arr = []
        key_arr = []
        
        for f_key in ml_dicts[m_key].keys():
            str_key = str(f_key)
            if str_key[0:3] == str(s):
                base_key = str(s)+', F1'
                f1_diff_arr.append( ml_dicts[m_key][f_key][ 'tot_f1_test' ] -  ml_dicts['lr'][base_key][ 'tot_f1_test' ]  )
                sd_train = jack_SD( ml_dicts['lr'][base_key][ 'jack_train'] ,  ml_dicts[m_key][f_key]['jack_train'] )[0]
                sd_test =  jack_SD( ml_dicts['lr'][base_key][ 'jack_test' ] ,  ml_dicts[m_key][f_key]['jack_test']  )[0]

                sd = np.sqrt( np.array((sd_train**2)) + np.array((sd_test**2)))

                sd_diff_arr.append(sd) # append sd_arr to an array
                key_arr.append(f_key)
                
        f1_diff_all.append([ key_arr, f1_diff_arr, sd_diff_arr]) 
    return f1_diff_all



def base_per_split_vald(ml_dicts, models, s):

    # List containing data for different classifiers
    f1_diff_all = []

    # Loop through models to complie all data
    for m, m_key in zip (models, ml_dicts.keys()):
        f1_diff_arr = [] 
        sd_diff_arr = []
        key_arr = []
        
        for f_key in ml_dicts[m_key].keys():
            str_key = str(f_key)
            if str_key[0:3] == str(s):
                base_key = str(s)+', F1'
                f1_diff_arr.append( ml_dicts[m_key][f_key][ 'tot_f1_vald' ] -  ml_dicts['lr'][base_key][ 'tot_f1_vald' ]  )
                sd_train = jack_SD( ml_dicts['lr'][base_key][ 'jack_train'] ,  ml_dicts[m_key][f_key]['jack_train'] )[0]
                sd_test =  jack_SD( ml_dicts['lr'][base_key][ 'jack_vald' ] ,  ml_dicts[m_key][f_key]['jack_vald']  )[0]

                sd = np.sqrt( np.array((sd_train**2)) + np.array((sd_test**2)))

                sd_diff_arr.append(sd) # append sd_arr to an array
                key_arr.append(f_key)
                
        f1_diff_all.append([ key_arr, f1_diff_arr, sd_diff_arr]) 
    return f1_diff_all