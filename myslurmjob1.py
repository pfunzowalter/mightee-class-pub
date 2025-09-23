                     # necessary modules
from sources.ml_f1 import*
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


# We call the test and train data saved from the processing notebook
mightee_data = pd.read_csv('raw_data.csv')

# The train data
X_data = pd.read_csv('X_train.csv')
y_data = pd.read_csv('y_train.csv')

# The Unseen test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

y = y_data['labels']

# Hyperparameters
# XGBOOST
#XGBoost hyper-parameter tuning
xgb_par = {'learning_rate': [0.01, 0.15, 0.23, 0.3],
                'max_depth': [3, 7, 10],
                'n_estimators' : [50, 100, 150, 200, 300]}

xgb_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='rmse', n_jobs=-1 )


## random forest (RF)
# The Random Hyper parameter Grid

# number of trees in the forest
n_estimators = [50, 100, 150, 200, 300, 450]

# Number of feature to consider at every split
max_features = ["sqrt", "log2", "NONE"]

# Maximum number of levels in tree
max_depth = [5, 10, 30, 40, 50, 60, 70]

# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3]

# Method of selecting samples for training each tree
bootstrap = [True, False]


# Create the random grid
rf_par = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



rf_model= RandomForestClassifier(random_state=1)
rf_par = dict(n_estimators=n_estimators)



# Support Vector Machines
svm_model = SVC()

# defining parameter range
svm_par = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid' ]}
# KNN model
knn_model = KNeighborsClassifier()

## KNN parameters
knn_par = {'n_neighbors' : [5, 10, 15],
           'p':[1, 2], 
           'weights' : ['uniform', 'distance'] }


## logisitc regression (LR)
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [1000, 100, 10, 1.0, 0.1, 0.01, 0.001]

lr_model = LogisticRegression()
lr_par = dict(solver=solvers,penalty=penalty,C=c_values)



# Set up models and Parameters for a "for loop"  

models = [[lr_model, 'lr'], [svm_model, 'svm'], [knn_model, 'knn'], [rf_model, 'rf'], [xgb_model, 'xgb']]

parameters = [ lr_par, svm_par, knn_par, rf_par, xgb_par]


ml_dicts = {}

features = [['qir'], 
            ['qir', 'class_star'],
            ['qir', 'class_star', 'log(S8/S45)'],
            ['qir', 'class_star', 'log(S8/S45)','log(S58/S36)'],
            ['qir', 'class_star', 'log(S8/S45)','log(S58/S36)', 'Mstar'],
            # ['qir', 'class_star', 'log(S8/S45)','log(S58/S36)', 'Mstar', 'log(S45/S36)'],
            # ['qir', 'class_star', 'Mstar', 'log(S45/S36)']
           ]

splits = [0.8, 0.6, 0.4, 0.2]
tr_sizes = [0.2, 0.4, 0.6, 0.8]
# Loop through different ML models coupled with thier hyper paramter (use the same splits for all features)
for m, par in zip(models, parameters):
    key0 = str(m[1])
    print(key0)
    ml_dicts[key0] = {} # defining The main subkeys, which are the machine learning models
        
    for s, tr in zip(splits, tr_sizes):
        X_train, X_vald, y_train, y_vald = train_test_split(X_data, y, test_size= s, random_state=1, stratify = y, shuffle = True)
        
        i = 1
        for f in features:
            xtr =  X_train[f]
            xva =  X_vald[f]
            xte =  X_test[f]
            
            results = get_f1_ml (m[0], par, xtr, y_train, xva, y_vald, xte, y_test) # to get the f1 for the ml model

            key = str(tr)+", F"+str((i)) # Create keys for the each feature set in order to reference results
            ml_dicts[key0][key] = {}

            ml_dicts[key0][key]['tot_f1_vald'] = results[0]
            ml_dicts[key0][key]['tot_f1_test'] = results[1]
            ml_dicts[key0][key]['jack_train'] = results[2]
            ml_dicts[key0][key]['jack_vald'] = results[3]
            ml_dicts[key0][key]['jack_test'] = results[4]
            i += 1
            
            
import json
# with open('final_catlogue_ml_data.txt', 'w') as file:
with open('normalised/final_review_data.txt', 'w') as file:
    file.write(json.dumps(ml_dicts)) 