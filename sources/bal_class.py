import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import timeit

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer

from sklearn.model_selection import train_test_split



# BALANCED CROSS VALIDATION

def balanced_cv( X, y, model, train_size, binary = True):
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    over_sampler = RandomOverSampler(random_state=42)
    under_sampler = RandomUnderSampler(random_state=42)
    
    
    indices = np.arange(X.shape[0])
    
    X_dummies = pd.get_dummies(X)
    
    if binary == True:
        #         Scores per class
        agn_f1_bal, sfg_f1_bal = [], []
        agn_rec_bal, sfg_rec_bal = [], []
        agn_pre_bal, sfg_pre_bal = [], []
        
        # Random State
        random_vals = np.arange(0, 500, 10)
        
        for i in range(len(train_size)):
            # define the test size
            test = 1 - train_size[i]
            # ramdomly spliting the data n times for each test-size
            agn_f1_random_bal, sfg_f1_random_bal = [], []
            agn_rec_random_bal, sfg_rec_random_bal = [], []
            agn_pre_random_bal, sfg_pre_random_bal = [], []
            n  = 0
            while n < 20:
            #     we split the sample into the testing and training
                X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size = test,
                                                                                      random_state = random_vals[i],
                                                                                      stratify = y)

                
#                 Resampling the SFGs to equal the galaxies
                X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    
            # Source Classification 
                model2 = model.fit(X_res, y_res) # balanced training
                
                y_xgb_bal = model2.predict(X_test) # prediction from a model trained on the balanced data
                
                agn_f1_sco1_bal  = f1(y_test, y_xgb_bal, average = 'weighted', labels = [0])
                sfg_f1_sco1_bal  = f1(y_test, y_xgb_bal, average = 'weighted', labels = [1])
                
    
                agn_rec1_bal = recall(y_test, y_xgb_bal, average = 'weighted', labels = [0])
                sfg_rec1_bal = recall(y_test, y_xgb_bal, average = 'weighted', labels = [1])

                agn_pre1_bal = precision(y_test, y_xgb_bal, average = 'weighted', labels = [0])
                sfg_pre1_bal = precision(y_test, y_xgb_bal, average = 'weighted', labels = [1])
        

                agn_f1_random_bal.append(agn_f1_sco1_bal)
                sfg_f1_random_bal.append(sfg_f1_sco1_bal)
        
                agn_rec_random_bal.append(agn_rec1_bal)
                sfg_rec_random_bal.append(sfg_rec1_bal)
                
                agn_pre_random_bal.append(agn_pre1_bal)
                sfg_pre_random_bal.append(sfg_pre1_bal)
                n = n + 1
            
            agn_f1_bal.append(np.mean(agn_f1_random_bal))
            sfg_f1_bal.append(np.mean(sfg_f1_random_bal))
            
            agn_rec_bal.append(np.mean(agn_rec_random_bal))
            sfg_rec_bal.append(np.mean(sfg_rec_random_bal))
                              
            agn_pre_bal.append(np.mean(agn_pre_random_bal))
            sfg_pre_bal.append(np.mean(sfg_pre_random_bal))
            
            #         converting scores to a dataframe
        agn_metrics = np.vstack([np.array(train_size),
                                 np.array(agn_f1_bal),
                                 np.array(agn_pre_bal),
                                 np.array(agn_rec_bal)]).T
    
    
        sfg_metrics = np.vstack([np.array(train_size),
                                 np.array(sfg_f1_bal),
                                 np.array(sfg_pre_bal),
                                 np.array(sfg_rec_bal)]).T
        
        agn_score_data = pd.DataFrame(agn_metrics, columns = ['train_size', 'f1', 'precision', 'recall'])
        sfg_score_data = pd.DataFrame(sfg_metrics, columns = ['train_size', 'f1', 'precision', 'recall'])
        
        return agn_score_data, sfg_score_data