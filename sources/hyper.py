from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Functions
def cv (m, p, xtrain, ytrain):
    inner_cv = StratifiedKFold(n_splits=3)
    clf = RandomizedSearchCV(m, 
                       p, 
                       scoring=make_scorer(f1), 
                       n_jobs=-1, 
                       cv=inner_cv, 
                       refit=True, 
                       verbose=0)
    clf.fit(xtrain, ytrain)
    
    return clf


# The train_VS_score manual cross validation
def hyper_ml_cv( model, parameter, X, y, train_size, binary = True):
    
    indices = np.arange(X.shape[0])
    
    X_dummies = pd.get_dummies(X)
    
    if binary == True:
        #         Scores per class
        f1_sco = []
        rec = []
        pre = []
        # Random State
        random_vals = np.arange(0, 500, 10)
        
        for i in range(len(train_size)):
            # define the test size
            test = 1 - train_size[i]
            # ramdomly spliting the data n times for each test-size
            f1_random = []
            rec_random = []
            pre_random = []
            n  = 0
            while n < 20:
            #     we split the sample into the testing and training
                X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size = test,
                                                                                      random_state = random_vals[i],
                                                                                      stratify = y, shuffle = True)

            # Source Classification
                clf = cv(model, parameter, X_train, y_train)  

                y_xgb = clf.predict(X_test)
                
                f1_sco1  = f1(y_test, y_xgb)
                
                
                rec1 = recall(y_test, y_xgb)

                pre1 = precision(y_test, y_xgb)
        

                f1_random.append(f1_sco1)
                
                rec_random.append(rec1)
                
                pre_random.append(pre1)
                n = n + 1

            f1_sco.append(np.mean(f1_random))
            
            rec.append(np.mean(rec_random))
            
            pre.append(np.mean(pre_random))



        
        #         converting scores to a dataframe
        score_metrics = np.vstack([np.array(train_size),
                                 np.array(f1_sco),
                                 np.array(pre),
                                 np.array(rec)]).T
        
        score_data = pd.DataFrame(score_metrics, columns = ['train_size', 'f1', 'precision', 'recall'])
        
        return score_data
        
    else:
        print('Binary classification only')