import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
# from sklearn.metrics import recall_score as recall
from sklearn.metrics import recall_score as recall
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

def custom_recall(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same.")
    
    matching_count = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    total_count = len(y_true)
    
    matching_percentage = (matching_count / total_count)
    return matching_percentage

def calculate_recall(y_true, y_pred):
    """
    Calculate the recall score manually.

    Parameters:
    y_true (list): A list of true labels.
    y_pred (list): A list of predicted labels.

    Returns:
    float: The recall score.
    """
    
    # Initialize counters for true positives and false negatives
    true_positives = 0
    false_negatives = 0
    
    # Iterate over the true and predicted labels
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            true_positives += 1
        elif true == 1 and pred == 0:
            false_negatives += 1
    
    # Calculate recall
    if true_positives + false_negatives == 0:
        return 0.0  # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives)
    
    return recall

# important functions
def cv (m, p, xtrain, ytrain, xVald, yVald):
    
    inner_cv = StratifiedKFold(n_splits=3)
    clf = GridSearchCV(m, 
                       p, 
                       scoring=make_scorer(recall), 
                       n_jobs=-1, 
                       cv=inner_cv, 
                       refit=True, 
                       verbose=0)
    
    clf.fit(X = np.array(xtrain), y=np.array(ytrain).reshape(len(ytrain)).ravel())
    
    yVald1 = np.array(yVald).reshape(len(yVald)).ravel()
    
    pred = clf.predict(xVald)
    
    return recall(yVald1, pred), clf


def get_recall_ml (m, p, xTrain, yTrain, xVald, yVald, xTest, yTest, reference_test ):
    print("This is reference", len(reference_test))
    recallTot_vald, clfTot_vald = cv(m, p, xTrain, yTrain, xVald, yVald)
    
    y_Tot_pred = clfTot_vald.predict(xTest)        
    ref_test_df = reference_test
    # ADDED NEW PARTS
    ref_test_df['class-ml'] = y_Tot_pred
    
    xray = ref_test_df[ref_test_df['XAGN'] == True].copy()
    vlbi = ref_test_df[ref_test_df['VLBAAGN'] == True].copy()

    y_xray, clas_xray = pd.factorize(xray['XAGN'])
    y_vlbi, clas_vlbi = pd.factorize(vlbi['VLBAAGN'])
    xray['XAGN_clas'] = y_xray
    vlbi['VLBAAGN_clas'] = y_vlbi
    
    # xray.to_csv('final-train-test/ml_xray_04_march'+str(m)+'.csv', index = False, header=True)
    # vlbi.to_csv('final-train-test/ml_vlbi_04_march'+str(m)+'.csv', index = False, header=True)
    
    # xray['XAGN'] = [0 if item is True else item for item in xray['XAGN']]
    # vlbi['VLBAAGN'] = [0 if item is True else item for item in vlbi['VLBAAGN']]
    print("This is test", len(yTest))
    recallTot_test = recall(yTest, y_Tot_pred)
    recallTot_xray = custom_recall(xray['XAGN_clas'], xray['class-ml'])
    recallTot_vlbi = custom_recall(vlbi['VLBAAGN_clas'], vlbi['class-ml'])
    print("This is predicted", len(y_Tot_pred))
    
     
    jackTrainArr = []
    jackValdArr = []
    jackTestArr = []
    jackxrayArr = []
    jackvlbiArr = []
            
    for i in range(0, len(xTrain), 10):
        x_train = np.delete(np.array(xTrain), i, 0)
        y_train = np.delete(np.array(yTrain), i, 0)
        
        scoreTrain, clrecall = cv(m, p, x_train, y_train, xVald, yVald)
        
        jackTrainArr.append(scoreTrain)

    for t in range (len(xVald)):
        x_vald = np.delete(np.array(xVald), t, 0)
        y_vald = np.delete(np.array(yVald), t, 0)
            
        y_predict = clrecall.predict(x_vald)
        vscore = recall(y_vald, y_predict)
        
        jackValdArr.append(vscore)
        
    for p in range (len(xTest)):
        x_test = np.delete(np.array(xTest), p, 0)
        y_test = np.delete(np.array(yTest), p, 0)

        test_df_1 = reference_test
        test_df_1 = test_df_1.drop(index=p)
            
        y_pred = clrecall.predict(x_test)
                
        test_df_1['class-ml'] = y_pred
        
        xray1 = test_df_1[test_df_1['XAGN'] == True].copy()
        vlbi1 = test_df_1[test_df_1['VLBAAGN'] == True].copy()

        y_xray1, clas_xray1 = pd.factorize(xray1['XAGN'])
        y_vlbi1, clas_vlbi1 = pd.factorize(vlbi1['VLBAAGN'])
        xray1['XAGN_clas'] = y_xray1
        vlbi1['VLBAAGN_clas'] = y_vlbi1
    
        # xray1['XAGN'] = [0 if item is True else item for item in xray1['XAGN']]
        # vlbi1['VLBAAGN'] = [0 if item is True else item for item in vlbi1['VLBAAGN']]
        
        tscore_xray = custom_recall(xray1['XAGN_clas'], xray1['class-ml'])
        tscore_vlbi = custom_recall(vlbi1['VLBAAGN_clas'], vlbi1['class-ml'])
        tscore = recall(y_test, y_pred)
        
        
        jackTestArr.append(tscore)  
        jackxrayArr.append(tscore_xray)  
        jackvlbiArr.append(tscore_vlbi)  
        
            
    return  recallTot_vald, recallTot_test, recallTot_xray, recallTot_vlbi, jackTrainArr, jackValdArr, jackTestArr, jackxrayArr, jackvlbiArr


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


def result_per_split_recall(ml_dicts,models, s):
    arr_all = []
    for m, d in zip (models, ml_dicts.keys()):
        f1_arr_vald = []
        f1_arr_test = []
        f1_arr_xray = []
        f1_arr_vlbi = []
        sd_vald_arr = []
        sd_arr = [] 
        sd_arr_vlbi = [] 
        sd_arr_xray = [] 
        
        # print(ml_dicts[d])
        for key in ml_dicts[d].keys():
            f1_arr_vald.append(ml_dicts[d][key][ 'tot_recall_vald' ]) # append total valdation f1 score to an array
            f1_arr_test.append(ml_dicts[d][key][ 'tot_recall_test' ]) # append total test f1 score to an array
            f1_arr_xray.append(ml_dicts[d][key][ 'tot_recall_xray' ]) # append total test f1 score to an array
            f1_arr_vlbi.append(ml_dicts[d][key][ 'tot_recall_vlbi' ]) # append total test f1 score to an array
            sd_train = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_train' ]) ), ml_dicts[d][key][ 'jack_train' ])[0]
            sd_vald = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_vald' ]) ), ml_dicts[d][key][ 'jack_vald' ])[0]
            sd_test = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_test' ]) ), ml_dicts[d][key][ 'jack_test' ])[0]
            sd_xray = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_xray' ]) ), ml_dicts[d][key][ 'jack_xray' ])[0]
            sd_vlbi = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_vlbi' ]) ), ml_dicts[d][key][ 'jack_vlbi' ])[0]
            
            
            sd_v = np.sqrt( np.array((sd_train**2)) + np.array((sd_vald**2)))
            sd = np.sqrt( np.array((sd_train**2)) + np.array((sd_test**2)))
            sd_xray = np.sqrt( np.array((sd_train**2)) + np.array((sd_xray**2)))
            sd_vlbi = np.sqrt( np.array((sd_train**2)) + np.array((sd_vlbi**2)))
           
            sd_vald_arr.append(sd_v)
            sd_arr.append(sd)
            sd_arr_vlbi.append(sd)
            sd_arr_xray.append(sd)
            # append the SD to the sd_arr
        arr_all.append([ list(ml_dicts[d].keys()), f1_arr_vald, f1_arr_test, f1_arr_xray, f1_arr_vlbi, sd_vald_arr, sd_arr, sd_arr_xray, sd_arr_vlbi])    
    
    return arr_all
    

def base_per_split(ml_dicts, models, s):

    # List containing data for different classifiers
    recall_diff_all = []

    # Loop through models to complie all data
    for m, m_key in zip (models, ml_dicts.keys()):
        recall_diff_arr = [] 
        sd_diff_arr = []
        key_arr = []
        
        for f_key in ml_dicts[m_key].keys():
            str_key = str(f_key)
            if str_key[0:3] == str(s):
                base_key = str(s)+', recall'
                recall_diff_arr.append( ml_dicts[m_key][f_key][ 'tot_recall_test' ] -  ml_dicts['lr'][base_key][ 'tot_recall_test' ]  )
                sd_train = jack_SD( ml_dicts['lr'][base_key][ 'jack_train'] ,  ml_dicts[m_key][f_key]['jack_train'] )[0]
                sd_test =  jack_SD( ml_dicts['lr'][base_key][ 'jack_test' ] ,  ml_dicts[m_key][f_key]['jack_test']  )[0]

                sd = np.sqrt( np.array((sd_train**2)) + np.array((sd_test**2)))

                sd_diff_arr.append(sd) # append sd_arr to an array
                key_arr.append(f_key)
                
        recall_diff_all.append([ key_arr, recall_diff_arr, sd_diff_arr]) 
    return recall_diff_all



def base_per_split_vald(ml_dicts, models, s):

    # List containing data for different classifiers
    recall_diff_all = []

    # Loop through models to complie all data
    for m, m_key in zip (models, ml_dicts.keys()):
        recall_diff_arr = [] 
        sd_diff_arr = []
        key_arr = []
        
        for f_key in ml_dicts[m_key].keys():
            str_key = str(f_key)
            if str_key[0:3] == str(s):
                base_key = str(s)+', recall'
                recall_diff_arr.append( ml_dicts[m_key][f_key][ 'tot_recall_vald' ] -  ml_dicts['lr'][base_key][ 'tot_recall_vald' ]  )
                sd_train = jack_SD( ml_dicts['lr'][base_key][ 'jack_train'] ,  ml_dicts[m_key][f_key]['jack_train'] )[0]
                sd_test =  jack_SD( ml_dicts['lr'][base_key][ 'jack_vald' ] ,  ml_dicts[m_key][f_key]['jack_vald']  )[0]

                sd = np.sqrt( np.array((sd_train**2)) + np.array((sd_test**2)))

                sd_diff_arr.append(sd) # append sd_arr to an array
                key_arr.append(f_key)
                
        recall_diff_all.append([ key_arr, recall_diff_arr, sd_diff_arr]) 
    return recall_diff_all