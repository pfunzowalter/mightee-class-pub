import numpy as np
import matplotlib.pyplot as pl
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import f1_score as f1
from sklearn.metrics import make_scorer
import random

def perm_impotance(X, y, model):
    iterations = np.arange(0, 5000, 50)
    mean_importances = []
    std_importances = []
    labels = []
    for i in range(100):
        random_tsize = random.uniform(0.5, 0.96)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            stratify=y, test_size = random_tsize,
                                                            random_state = iterations[i])
        
        model.fit(X_train, y_train)

        result = permutation_importance(
            model, X_train, y_train, n_repeats=10, random_state=42, scoring = make_scorer(f1), n_jobs=2)
                              
        mean_importances.append(result.importances_mean)
        std_importances.append(result.importances_std)
        
    return mean_importances, std_importances


def manual_mean_importances(importances):

    ft1, ft2,  ft3, ft4, ft5, ft6 = [], [], [], [], [], []

    for m in range(len(importances)):
        ft1.append(importances[m][0])
        ft2.append(importances[m][1])
        ft3.append(importances[m][2])
        ft4.append(importances[m][3])
        ft5.append(importances[m][4])
        ft6.append(importances[m][5])
    ft1_mean, ft2_mean, ft3_mean = np.mean(ft1),np.mean(ft2),np.mean(ft3)
    ft4_mean, ft5_mean, ft6_mean  = np.mean(ft4),np.mean(ft5),np.mean(ft6)
    array = np.array([ft1_mean, ft2_mean, ft3_mean, ft4_mean, ft5_mean, ft6_mean])
    
    return array
