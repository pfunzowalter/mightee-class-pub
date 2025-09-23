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
import json
# plot style
import seaborn as sns
import timeit
sns.set_style("whitegrid")#, {"axes.facecolor": ".9"})


# def cross_val(clf, X, y):
#     scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
#     return np.mean(scores), np.std(scores)
def classifier(model, X_train, y_train, X_test, y_test):
    scores = []
    score_err = []
    
    start_time = timeit.default_timer()
    # for i in range(10):
        # Source Classification

        # model.fit(X_train, y_train)  

        # y_pred = model.predict(X_test)

        # # proba = model.predict_proba(X_test)

        # score = f1(y_test, y_pred)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')  # 5-fold cross-validation
        
        # scores.append(score)
        
    
    elapsed = timeit.default_timer() - start_time
    
    # print('Elapsed time for classifier: {} seconds'.format(elapsed))
    # print(len(y_pred))
    print('F1 for classifier is: {}'.format(scores.mean()))#np.mean(scores)))
    print('F1 error for classifier is: {}'.format(scores.std()))
    
    # print(metrics.classification_report(y_test, y_xgb, target_names=['AGN', "SFG"], digits=4))
        
    # return np.mean(scores), np.std(scores)
    return scores.mean(), scores.std()

def plot_par_scores(par_name, params, model_name, scores, score_err, text, ylmt=[]):
    print("this us a test: ", str(par_name) +' of '+ str(model_name)) 
    if type(params[0])== str:
        x_pos = np.arange(len(params))   
        # Build the plot
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 12)
        ax.bar(x_pos, scores, yerr=score_err, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('F1-Scores', fontweight ='bold', fontsize =32)
        ax.set_xticks(x_pos)
        if model_name != "knn":
            ax.set_xticklabels(par_name +' of '+ model_name.upper(), fontweight ='bold', fontsize =32)
        else:
            ax.set_xticklabels(par_name +' of '+ '$k$NN', fontweight ='bold', fontsize =32)
        ax.yaxis.grid(True)
        ax.set_ylim(0.8, 0.98)
        plt.text(0.07, 0.97, text, 
                 transform=plt.gca().transAxes, fontsize=30, 
                 bbox=dict(
                     facecolor='white',
                     edgecolor='0.8',  # Default legend frame color (light gray)
                     alpha=1.0,       # Fully opaque (default)
                     boxstyle='round',  # Matches default legend rounding
                     linewidth=0.8,    # Default legend frame linewidth
                 ),
                 verticalalignment='top', 
                 horizontalalignment='right')
        # Save the figure and show
        plt.tight_layout()
        plt.savefig(par_name, bbox_inches='tight')
        plt.show()
            
    else:
        plt.figure(figsize = (16,12))
        # plt.figure(figsize = (10, 8))
        plt.errorbar(x = params,  y = scores, yerr = score_err, fmt="o", color="b")
        if model_name != "knn":
            plt.xlabel(par_name +' of '+ model_name.upper(), fontweight ='bold', fontsize =36)
        else:
            plt.xlabel(par_name +' of '+ '$k$NN', fontweight ='bold', fontsize =36)
            
        # plt.xlabel(par_name +' of '+ model_name.upper(), fontweight ='bold', fontsize =36)
        plt.ylabel('F1-score', fontweight ='bold', fontsize =36)
        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
                # Enable minor ticks
        plt.minorticks_on()
        
        # Add major and minor grid
        plt.text(0.07, 0.97, text, 
         transform=plt.gca().transAxes, fontsize=30,
         bbox=dict(
                     facecolor='white',
                     edgecolor='0.8',  # Default legend frame color (light gray)
                     alpha=1.0,       # Fully opaque (default)
                     boxstyle='round',  # Matches default legend rounding
                     linewidth=0.8,    # Default legend frame linewidth
                 ),
                 verticalalignment='top', 
                 horizontalalignment='right')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.ylim(ylmt[0], ylmt[1])
        plt.savefig("plots/hyperparameters/"+par_name.upper()+'.pdf', bbox_inches='tight')
        plt.show()
    
def hyper_par( X, y, parameters, par_name = '', model_name = '', txt='', ylmt = [] ):
    text = txt
    lr_keys = ['solvers', 'penalty', 'C' ]
    knn_keys = ['n_neighbors', 'p', 'weights']
    svm_keys = ['C', 'gamma', 'kernel']
    rf_keys = [ 'n_estimators', 'max_features', 'max_depth', 'min_samples_split', 
               'min_samples_leaf',
               'bootstrap']
    xgb_keys = ['learning_rate',
                'max_depth',
                'n_estimators']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25, random_state=42)
    
    # LR 
    if model_name == 'lr' and par_name in lr_keys:
        scores1 = []
        score_err1 = []
        
        i = 0
        print(len(parameters))
        while i < (len(parameters)):
            
            if par_name == 'solvers':
                params = {'solvers': parameters[i]}
                clf = LogisticRegression( **params)
            elif par_name == 'penalty':
                params = {'penalty': parameters[i]}
                clf = LogisticRegression(**params)
            elif par_name == 'C':
                params = {'C': parameters[i]}
                clf = LogisticRegression(**params)
            else:
                print('Parameter not found, using default parameters')
                clf = LogisticRegression()
            
            score, err = classifier(clf, X_train, y_train, X_test, y_test)
            
            print('current parameter is :', par_name, parameters[i])
            
            scores1.append(score)
            score_err1.append(err)
            
            i += 1
        scores = np.array(scores1)
        score_err = np.array(score_err1)  
        
        print("Model scores", scores)
        print("Model scores errors", score_err)
        
        # df = pd.DataFrame([scores, score_err, params], columns = ['f1', 'f1_err', par_name])
            
        plot_par_scores(par_name, parameters, model_name, scores, score_err, text, ylmt=ylmt)
    
                                                        
    # KNN   
    elif model_name == 'knn' and par_name in knn_keys:
    
        scores = []
        score_err = []
        
        for i in range(len(parameters)):
            if par_name == 'n_neighbors':
                params = {'n_neighbors': parameters[i]}
                clf = KNeighborsClassifier(**params)
            elif par_name == 'p':
                params = {'p': parameters[i]}
                clf = KNeighborsClassifier(**params)
            elif par_name == 'weights':
                params = {'weights': parameters[i]}
                clf = KNeighborsClassifier(**params)
            else:
                print('Parameter not found, using default parameters')
                clf = KNeighborsClassifier()
                
            score, err = classifier(clf, X_train, y_train, X_test, y_test)
            
            scores.append(score)
            score_err.append(err)
        
        scores = np.array(scores)
        score_err = np.array(score_err)  
        
        # df = pd.DataFrame([scores, score_err, params], columns = ['f1', 'f1_err', par_name])
        print("this is ylims", ylmt)
        print("this is first ylims", ylmt[0])    
        print("this is second ylims", ylmt[1])    
        
        # plot_par_scores(par_name, parameters, scores, score_err, ylim=ylim)
        plot_par_scores(par_name, parameters, model_name, scores, score_err, text, ylmt=ylmt)
        
                                                        
    
    # SVM   
    elif model_name == 'svm' and par_name in svm_keys:
    
        scores = []
        score_err = []
        
        for i in range(len(parameters)):
            if par_name == 'C':
                params = {'C': parameters[i]}
                clf = SVC(**params)
            elif par_name == 'gamma':
                params = {'gamma': parameters[i]}
                clf = SVC(**params)
            elif par_name == 'kernel':
                params = {'kernel': parameters[i]}
                clf = SVC(**params)
            else:
                print('Parameter not found, using default parameters')
                clf = SVC()
            
            score, err = classifier(clf, X_train, y_train, X_test, y_test)
            
            scores.append(score)
            score_err.append(err)
        
        scores = np.array(scores)
        score_err = np.array(score_err)  
        
        # df = pd.DataFrame([scores, score_err, params], columns = ['f1', 'f1_err', par_name])
            
        # plot_par_scores(par_name, parameters, scores, score_err) 
        plot_par_scores(par_name, parameters, model_name, scores, score_err, text, ylmt=ylmt)
        
                                                        
    # RF   
    elif model_name == 'rf' and par_name in rf_keys:
    
        scores = []
        score_err = []
        
        for i in range(len(parameters)):
            
            if par_name == 'n_estimators':
                params = {'n_estimators': parameters[i]}
                clf = RandomForestClassifier(**params)
            elif par_name == 'max_features':
                params = {'max_features': parameters[i]}
                clf = RandomForestClassifier(**params)
            elif par_name == 'max_depth':
                params = {'max_depth': parameters[i]}
                clf = RandomForestClassifier(**params)
            elif par_name == 'min_samples_split':
                params = {'min_samples_split': parameters[i]}
                clf = RandomForestClassifier(**params)
            elif par_name == 'min_samples_leaf':
                params = {'min_samples_leaf': parameters[i]}
                clf = RandomForestClassifier(**params)
            elif par_name == 'bootstrap':
                params = {'bootstrap': parameters[i]}
                clf = RandomForestClassifier(**params)
            else:
                print('Parameter not found, using default parameters')
                clf = RandomForestClassifier(random_state=1)
            
            score, err = classifier(clf, X_train, y_train, X_test, y_test)
            
            scores.append(score)
            score_err.append(err)
        
        scores = np.array(scores)
        score_err = np.array(score_err)  
        
        # df = pd.DataFrame([scores, score_err, params], columns = ['f1', 'f1_err', par_name])
            
        # plot_par_scores(par_name, parameters, scores, score_err)
        plot_par_scores(par_name, parameters, model_name, scores, score_err, text, ylmt=ylmt)
        
                                                        
                                                        
    # RF   
    elif model_name == 'xgb' and par_name in xgb_keys:
    
        scores = []
        score_err = []
        
        for i in range(len(parameters)):
            if par_name == 'learning_rate':
                params = {'use_label_encoder': False, 'eval_metric' : 'rmse', 'n_jobs' : -1 , 'learning_rate': parameters[i]}
                clf =  xgboost.XGBClassifier(**params)
            elif par_name == 'max_depth':
                params = {'use_label_encoder': False, 'eval_metric' : 'rmse', 'n_jobs' : -1 , 'max_depth': parameters[i]}
                clf =  xgboost.XGBClassifier(**params)
            elif par_name == 'n_estimators':
                params = {'use_label_encoder': False, 'eval_metric' : 'rmse', 'n_jobs' : -1 , 'n_estimators': parameters[i]}
                clf =  xgboost.XGBClassifier(**params)
                
            else:
                print('Parameter not found, using default parameters')
                clf = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='rmse', random_state=1)
            
            score, err = classifier(clf, X_train, y_train, X_test, y_test)
            
            scores.append(score)
            score_err.append(err)
        
        scores = np.array(scores)
        score_err = np.array(score_err)  
        
        # df = pd.DataFrame([scores, score_err, params], columns = ['f1', 'f1_err', par_name])
            
        # plot_par_scores(par_name, parameters, scores, score_err)
        plot_par_scores(par_name, parameters, model_name, scores, score_err, text, ylmt=ylmt)
        
                                                        
    else:
        print('either model %s not found or parameted %s not defined' % (model_name, par_name))
        
