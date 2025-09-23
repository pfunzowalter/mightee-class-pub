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

# Simple Machine Learning
def classifier(model, X_features, y):
    #     getting the indices
    indices = np.arange(X_features.shape[0])
    
    X_dummies = pd.get_dummies(X_features)
    
    #     we split the sample into the testing and training
    x_train, x_test, y_train, y_test, i_train, i_test = train_test_split(X_dummies, y, indices, test_size=0.20)
    
    # Source Classification
    start_time = timeit.default_timer()

    model.fit(x_train, y_train)  

    y_xgb = model.predict(x_test)

    elapsed = timeit.default_timer() - start_time

    proba = model.predict_proba(x_test)


    acu = accuracy(y_test, y_xgb)
    
    print('Elapsed time for XGB: {} seconds'.format(elapsed))
    print(len(y_xgb))
    print('Accuracy for XGB is: {}'.format(acu))
    print(metrics.classification_report(y_test, y_xgb, target_names=['AGN', "SFG"], digits=4))
    
def confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pl.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    pl.imshow(cm, interpolation='nearest', cmap=cmap)
    pl.title(title)
    pl.colorbar()
    tick_marks = np.arange(len(classes))
    pl.xticks(tick_marks, classes, rotation=45)
    pl.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pl.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pl.tight_layout()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()# Feature importance for the experiment
    
# Feature importance for the experiment
def feature_importance(data):
    importances = pd.DataFrame({
        'Feature': data.drop('AGN', axis=1).columns,
        'Importance': xgb_model.feature_importances_
    })
    importances = importances.sort_values(by='Importance', ascending=False)
    importances = importances.set_index('Feature')
    print(importances)
    
    pl.figure(figsize = (16, 10))

    importances.plot.bar()
    pl.show()
    
    
    

# The train_VS_score manual cross validation
def train_vs_score_cv( X, y, model, train_size, binary = True):
    
    indices = np.arange(X.shape[0])
    
    X_dummies = pd.get_dummies(X)
    
    if binary == True:
        #         Scores per class
        agn_f1, sfg_f1 = [], []
        agn_rec, sfg_rec = [], []
        agn_pre, sfg_pre = [], []
        # Random State
        random_vals = np.arange(0, 500, 10)
        
        for i in range(len(train_size)):
            # define the test size
            test = 1 - train_size[i]
            # ramdomly spliting the data n times for each test-size
            agn_f1_random, sfg_f1_random = [], []
            agn_rec_random, sfg_rec_random = [], []
            agn_pre_random, sfg_pre_random = [], []
            n  = 0
            while n < 20:
            #     we split the sample into the testing and training
                X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size = test,
                                                                                      random_state = random_vals[i],
                                                                                      stratify = y)

            # Source Classification
                model.fit(X_train, y_train)  

                y_xgb = model.predict(X_test)
                
                agn_f1_sco1  = f1(y_test, y_xgb, average = 'weighted', labels = [0])
                sfg_f1_sco1  = f1(y_test, y_xgb, average = 'weighted', labels = [1])
                
                
                agn_rec1 = recall(y_test, y_xgb, average = 'weighted', labels = [0])
                sfg_rec1 = recall(y_test, y_xgb, average = 'weighted', labels = [1])

                agn_pre1 = precision(y_test, y_xgb, average = 'weighted', labels = [0])
                sfg_pre1 = precision(y_test, y_xgb, average = 'weighted', labels = [1])
        

                agn_f1_random.append(agn_f1_sco1)
                sfg_f1_random.append(sfg_f1_sco1)
                
                agn_rec_random.append(agn_rec1)
                sfg_rec_random.append(sfg_rec1)
                
                agn_pre_random.append(agn_pre1)
                sfg_pre_random.append(sfg_pre1)
                n = n + 1

            agn_f1.append(np.mean(agn_f1_random))
            sfg_f1.append(np.mean(sfg_f1_random))
            
            agn_rec.append(np.mean(agn_rec_random))
            sfg_rec.append(np.mean(sfg_rec_random))
            
            agn_pre.append(np.mean(agn_pre_random))
            sfg_pre.append(np.mean(sfg_pre_random))
        
        #         converting scores to a dataframe
        agn_metrics = np.vstack([np.array(train_size),
                                 np.array(agn_f1),
                                 np.array(agn_pre),
                                 np.array(agn_rec)]).T
    
    
        sfg_metrics = np.vstack([np.array(train_size),
                                 np.array(sfg_f1),
                                 np.array(sfg_pre),
                                 np.array(sfg_rec)]).T
        
        agn_score_data = pd.DataFrame(agn_metrics, columns = ['train_size', 'f1', 'precision', 'recall'])
        sfg_score_data = pd.DataFrame(sfg_metrics, columns = ['train_size', 'f1', 'precision', 'recall'])
        
        return agn_score_data, sfg_score_data
        
    else:
         #         Scores per class
        agn_f1, sfg_f1, noclass_f1 = [], [], []
        agn_rec, sfg_rec, noclass_rec = [], [], []
        agn_pre, sfg_pre, noclass_pre = [], [], []
        for i in range(len(train_size)):
            # define the test size
            test_size = 1 - train_size[i]
            # ramdomly spliting the data n times for each test-size
            agn_f1_random, sfg_f1_random, noclass_f1_random = [], [], []
            agn_rec_random, sfg_rec_random, noclass_rec_random = [], [], []
            agn_pre_random, sfg_pre_random, noclass_pre_random = [], [], []
            n  = 0
            while n < 20:
            #     we split the sample into the testing and training
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=1997)

            # Source Classification
                model.fit(X_train, y_train)  

                y_xgb = model.predict(X_test)
                
                agn_f1_sco1  = f1(y_test, y_xgb, average = 'weighted', labels = ['AGN'])
                sfg_f1_sco1  = f1(y_test, y_xgb, average = 'weighted', labels = ['SFG'])
                noclass_f1_sco1  = f1(y_test, y_xgb, average = 'weighted', labels = ['noclass'])
                
                agn_rec1 = recall(y_test, y_xgb, average = 'weighted', labels = ['AGN'])
                sfg_rec1 = recall(y_test, y_xgb, average = 'weighted', labels = ['SFG'])
                noclass_rec1 = recall(y_test, y_xgb, average = 'weighted', labels = ['noclass'])

                agn_pre1 = precision(y_test, y_xgb, average = 'weighted', labels = ['AGN'])
                sfg_pre1 = precision(y_test, y_xgb, average = 'weighted', labels = ['SFG'])
                noclass_pre1 = precision(y_test, y_xgb, average = 'weighted', labels = ['noclass'])
        

                agn_f1_random.append(agn_f1_sco1)
                sfg_f1_random.append(sfg_f1_sco1)
                noclass_f1_random.append(noclass_f1_sco1)
                
                agn_rec_random.append(agn_rec1)
                sfg_rec_random.append(sfg_rec1)
                noclass_rec_random.append(noclass_rec1)
                
                agn_pre_random.append(agn_pre1)
                sfg_pre_random.append(sfg_pre1)
                noclass_pre_random.append(noclass_pre1)
                
                n = n + 1

            agn_f1.append(np.mean(agn_f1_random))
            sfg_f1.append(np.mean(sfg_f1_random))
            noclass_f1.append(np.mean(noclass_f1_random))
            
            agn_rec.append(np.mean(agn_rec_random))
            sfg_rec.append(np.mean(sfg_rec_random))
            noclass_rec.append(np.mean(noclass_rec_random))
            
            agn_pre.append(np.mean(agn_pre_random))
            sfg_pre.append(np.mean(sfg_pre_random))
            noclass_pre.append(np.mean(noclass_pre_random))


        pl.figure(figsize = (10, 12))

        #     F1
        pl.subplot(313)
        pl.plot(train_size, agn_f1,'--', label = 'AGN', c = 'r', linewidth = 3, alpha = 1)
        pl.plot(train_size, sfg_f1, '--', label = 'SFG', c = 'g', linewidth = 3, alpha = 1)
        pl.plot(train_size, noclass_f1, '--', label = 'noclass', c = 'g', linewidth = 3, alpha = 1)
        pl.axvline(0.2, c = 'y')
        pl.ylabel('F1')
        pl.xlabel('train size')
#         pl.title('F1 Scores per class')
        pl.legend(loc = 'lower right')

        #     Precision
        pl.subplot(311)
        pl.plot(train_size, agn_pre, '--', c = 'r', label = 'AGN', linewidth = 3, alpha = 1)
        pl.plot(train_size, sfg_pre, '--', c = 'g', label = 'SFG', linewidth = 3, alpha = 1)
        pl.plot(train_size, noclass_pre, '--', label = 'noclass', c = 'g', linewidth = 3, alpha = 1)
        pl.axvline(0.2, c = 'y')
        pl.ylabel('precision')
        pl.xlabel('train size')
#         pl.title('Precision Scores per class')
        pl.legend(loc = 'lower right')

        #   Recall
        pl.subplot(312)
        pl.plot(train_size, agn_rec, '--', c = 'r', label = 'AGN', linewidth = 3, alpha = 1)
        pl.plot(train_size, sfg_rec, '--', c = 'g', label = 'SFG', linewidth = 3, alpha = 1)
        pl.plot(train_size, noclass_rec, '--', label = 'noclass', c = 'g', linewidth = 3, alpha = 1)
        pl.axvline(0.2, c = 'y')
        pl.ylabel('recall')
        pl.xlabel('train size')
        pl.title('Recall Scores per class')
        pl.legend(loc = 'lower right')

        pl.savefig('train_scores')
        pl.show()
        
                #         converting scores to a dataframe
        agn_metrics = np.vstack([np.array(train_size),
                                 np.array(agn_f1),
                                 np.array(agn_pre),
                                 np.array(agn_rec)]).T
    
    
        sfg_metrics = np.vstack([np.array(train_size),
                                 np.array(sfg_f1),
                                 np.array(sfg_pre),
                                 np.array(sfg_rec)]).T
        
        noclass_metrics = np.vstack([np.array(train_size),
                                 np.array(noclass_f1),
                                 np.array(noclass_pre),
                                 np.array(noclass_rec)]).T
        
        agn_score_data = pd.DataFrame(agn_metrics, columns = ['train_size', 'f1', 'precision', 'recall'])
        sfg_score_data = pd.DataFrame(sfg_metrics, columns = ['train_size', 'f1', 'precision', 'recall'])
        noclass_score_data = pd.DataFrame(noclass_metrics, columns = ['train_size', 'f1', 'precision', 'recall'])
        
        return agn_score_data, sfg_score_data, noclass_score_data