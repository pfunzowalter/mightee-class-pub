#!/usr/bin/env python
from sources.ml_recall import*

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import pandas as pd

# ML models  
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from xgboost import plot_importance

X_data = pd.read_csv('final-train-test/X_train.csv')
y_data = pd.read_csv('final-train-test/y_train.csv')

# The Unseen test data
X_test = pd.read_csv('final-train-test/X_test.csv')
y_test = pd.read_csv('final-train-test/y_test.csv')

referece_test = pd.read_csv('final-train-test/original_test_df.csv')


y = y_data['labels']

# drop catid
X_data = X_data.drop(["CATID"], axis='columns')
X_test = X_test.drop(["CATID"], axis='columns')

print("The length of referece test", len(referece_test))
print("The length of Xtest", len(X_test))

# ### Hyperparameters
## random forest (RF)
# The Random Hyper parameter Grid

# number of trees in the forest
n_estimators = [50, 100, 150]

# Number of feature to consider at every split
max_features = [2, 3]

# Maximum number of levels in tree
max_depth = [5, 10]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3]

# Method of selecting samples for training each tree
bootstrap = [True, False]


# Create the random grid
rf_par = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth}
               # 'min_samples_split': min_samples_split,
               # 'min_samples_leaf': min_samples_leaf,
               # 'bootstrap': bootstrap}



rf_model= RandomForestClassifier(random_state=1)
rf_par = dict(n_estimators=n_estimators)

# Super Vector Machines
svm_model = SVC(kernel='linear')


svm_par = {'gamma': np.linspace(0.0001, 10, 15)}

# KNN model
knn_model = KNeighborsClassifier()

## KNN parameters
knn_par = {'n_neighbors' : [5, 10, 15], 'p':[1, 2], 'weights' : ['uniform', 'distance'] }


## logisitc regression (LR)
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [1000, 100, 10, 1.0, 0.1, 0.01, 0.001]

lr_model = LogisticRegression()
lr_par = dict(solver=solvers,penalty=penalty,C=c_values)

# Hyperparameters
# XGBOOST
#XGBoost hyper-parameter tuning
xgb_par = {'learning_rate': [0.01, 0.15, 0.23, 0.3],
                'max_depth': [3, 7, 10],
                'n_estimators' : [50, 100, 150, 200, 300]}

xgb_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='rmse', n_jobs=-1 )

# Set up models and Parameters for a "for loop"  
models = [[lr_model, 'lr'], [svm_model, 'svm'], [knn_model, 'knn'], [rf_model, 'rf'], [xgb_model, 'xgb']]

parameters = [ lr_par, svm_par, knn_par, rf_par, xgb_par]

ml_dicts = {}

features = [['qir', 'class_star', 'log(S8/S45)','log(S58/S36)', 'Mstar', 'log(S45/S36)']]

splits = [0.2] #, 0.4, 0.8]

# Loop through different ML models coupled with thier hyper paramter (use the same splits for all features)
for m, par in zip(models, parameters):
    for s in splits:
        # X_train, X_vald, y_train, y_vald = train_test_split(X_train_tb_noindex, y_train_tb, stratify = y_train_tb, test_size= s, random_state=1, shuffle = True)
        X_train, X_vald, y_train, y_vald = train_test_split(X_data, y, stratify = y, test_size= s, random_state=1, shuffle = True)
        
        key0 = str(m[1])
        print(key0)
        ml_dicts[key0] = {} # defining The main subkeys, which are the machine learning models
        
        i = 1
        for f in features:
            xtr =  X_train[f]
            xva =  X_vald[f]
            xte =  X_test[f]
            
            results = get_recall_ml (m[0], par, xtr, y_train, xva, y_vald, xte, y_test, referece_test) # to get the f1 for the ml model

            key = "F"+str((i)) # Create keys for the each feature set in order to reference results
            ml_dicts[key0][key] = {}

            ml_dicts[key0][key]['tot_recall_vald'] = results[0]
            ml_dicts[key0][key]['tot_recall_test'] = results[1]
            ml_dicts[key0][key]['tot_recall_xray'] = results[2]
            ml_dicts[key0][key]['tot_recall_vlbi'] = results[3]
            ml_dicts[key0][key]['jack_train'] = results[4]
            ml_dicts[key0][key]['jack_vald'] = results[5]
            ml_dicts[key0][key]['jack_test'] = results[6]
            ml_dicts[key0][key]['jack_xray'] = results[7]
            ml_dicts[key0][key]['jack_vlbi'] = results[8]
            
            i += 1


# In[ ]:
import json
with open('final-train-test/xray_vlbi_0.2_trainsize.txt', 'w') as file:
    file.write(json.dumps(ml_dicts)) 

ml_dicts.keys()


# In[ ]:


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
        sd_arr_xray.append(sd_xray)
        sd_arr_vlbi.append(sd_vlbi)
        # append the SD to the sd_arr
    arr_all.append([ list(ml_dicts[d].keys()), f1_arr_vald, f1_arr_test, f1_arr_xray, f1_arr_vlbi, sd_vald_arr, sd_arr, sd_arr_xray, sd_arr_vlbi])    

colors = ['blue', 'green', 'orange', 'red', 'purple']


def create_f1_score_dataframe(arr_all, models, features, colors):
    data = []
    count = 0
    n = 5

    for result, model, color in zip(arr_all, models, colors):
        a = np.linspace(n * count, n * (1 + count) - 2, len(features))  # to get index on the x-axis
        data.append({
            # 'Feature': feature,
            'Model': model[1],
            'recall_xray': result[3],
            'recall_vlbi': result[4],
            'xray Error': result[7],
            'vlbi Error': result[8],
            # 'Color': color
        })
        count += 1

    df = pd.DataFrame(data)
    df.to_csv('final-train-test/xray_vlbi_0.8_trainsize_job_2025_03_19.csv', index = False, header=True)
    return df

create_f1_score_dataframe(arr_all, models, features, colors)



