#!/usr/bin/env python
from sources.ml_f1 import*
from sources.ml_precision import*
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

# # The train data
X_train1 = pd.read_csv('X_train.csv')
y_train1 = pd.read_csv('y_train.csv')

# # The Unseen test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')


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


# #### SVM
# Super Vector Machines
svm_model = SVC(kernel='linear')


svm_par = {'gamma': np.linspace(0.0001, 10, 15)}


# #### KNN and LR
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

# XGBOOST
#XGBoost hyper-parameter tuning
xgb_par = {'learning_rate': [0.01, 0.15, 0.23, 0.3],
                'max_depth': [3, 7, 10],
                'n_estimators' : [50, 100, 150, 200, 300]}

xgb_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='rmse', n_jobs=-1 )


models = [[lr_model, 'lr'], [svm_model, 'svm'], [knn_model, 'knn'], [rf_model, 'rf'], [xgb_model, 'xgb']]
parameters = [ lr_par, svm_par, knn_par, rf_par, xgb_par]

ml_dicts = {}

features = [['qir', 'class_star', 'log(S8/S45)','log(S58/S36)', 'Mstar', 'log(S45/S36)']]

splits = [0.2]

# Loop through different ML models coupled with thier hyper paramter (use the same splits for all features)
for m, par in zip(models, parameters):
    for s in splits:
        X_train, X_vald, y_train, y_vald = train_test_split(X_train1, y_train1, stratify = y_train1, test_size= s, random_state=1, shuffle = True)
        # X_train, X_vald, y_train, y_vald = train_test_split(X_train, y_train, stratify = y_train, test_size= s, random_state=1, shuffle = True)
        key0 = str(m[1])
        print(key0)
        ml_dicts[key0] = {} # defining The main subkeys, which are the machine learning models
        
        i = 1
        for f in features:
            xtr =  X_train[f]
            xva =  X_vald[f]
            xte =  X_test[f]
            
            results = get_recall_ml (m[0], par, xtr, y_train, xva, y_vald, xte, y_test) # to get the f1 for the ml model
            # results = get_f1_ml (m[0], par, xtr, y_train, xva, y_vald, xte, y_test) # to get the f1 for the ml model

            key = "F"+str((i)) # Create keys for the each feature set in order to reference results
            ml_dicts[key0][key] = {}

            ml_dicts[key0][key]['tot_f1_vald'] = results[0]
            ml_dicts[key0][key]['tot_f1_test'] = results[1]
            ml_dicts[key0][key]['jack_train'] = results[2]
            ml_dicts[key0][key]['jack_vald'] = results[3]
            ml_dicts[key0][key]['jack_test'] = results[4]
            i += 1


# In[ ]:


arr_all = []
for m, d in zip (models, ml_dicts.keys()):
    f1_arr_vald = []
    f1_arr_test = []
    sd_vald_arr = []
    sd_arr = [] 
    
    # print(ml_dicts[d])
    for key in ml_dicts[d].keys():
        f1_arr_vald.append(ml_dicts[d][key][ 'tot_f1_vald' ]) # append total valdation f1 score to an array
        f1_arr_test.append(ml_dicts[d][key][ 'tot_f1_test' ]) # append total test f1 score to an array
        
        sd_train = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_train' ]) ), ml_dicts[d][key][ 'jack_train' ])[0]
        sd_vald = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_vald' ]) ), ml_dicts[d][key][ 'jack_vald' ])[0]
        sd_test = jack_SD(np.zeros( len(ml_dicts[d][key][ 'jack_test' ]) ), ml_dicts[d][key][ 'jack_test' ])[0]
        
        sd_v = np.sqrt( np.array((sd_train**2)) + np.array((sd_vald**2)))
        sd = np.sqrt( np.array((sd_train**2)) + np.array((sd_test**2)))
       
        sd_vald_arr.append(sd_v)
        sd_arr.append(sd)
        # append the SD to the sd_arr
    arr_all.append([ list(ml_dicts[d].keys()), f1_arr_vald, f1_arr_test, sd_vald_arr, sd_arr])    

# Assuming arr_all, models, and features are defined elsewhere in your code
colors = ['blue', 'green', 'orange', 'red', 'purple']

# Create a list to store the data
data = []

# Iterate through the results and models
for result, model in zip(arr_all, models):
    for f1_score, std_dev in zip(result[1], result[3]):
        data.append({
            'Model': model[1],
            'F1 Score': f1_score,
            'Standard Deviation': std_dev
        })

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Print the table
print(df)
# Save the table to a CSV file
# df.to_csv('notebook_recall_scores_table.csv', index=False)

import json
with open('scores/recall_0.2.txt', 'w') as file:
    file.write(json.dumps(ml_dicts)) 

