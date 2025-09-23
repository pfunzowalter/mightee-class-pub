import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision

from sklearn.model_selection import train_test_split
from scipy.spatial import distance


def measurments (data):
    cov = np.cov(data , rowvar=False)
    v1 = np.linalg.matrix_power(cov, -1)
    center = np.mean(data , axis=0)
    return cov, v1, center

def measurments1D (sample, data):
    #z scores of the points
    sd = np.std(data)
    center = np.mean(data)
    z = np.abs( (sample-center)/sd )
    return z



def nested (xtrain, ytrain, xtest, ytest, numFeat):
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain).flatten()
    xtest = np.array(xtest)
    ytest = np.array(ytest).flatten()

    
    xtrain_fer = xtrain[np.where(ytrain == 1)]
    xtrain_nf = xtrain[np.where(ytrain == 0)]
    testFer = []
    testNF = []
    
    if numFeat == 1:
        for test in xtest:
            testFer.append(measurments1D(test, xtrain_fer))
            testNF.append(measurments1D(test, xtrain_nf))
    
    else:    
        covFer, v1Fer, centerFer =  measurments(xtrain_fer)
        covNF, v1NF, centerNF =  measurments(xtrain_nf)
        
        for test in xtest:
            testFer.append(distance.mahalanobis(test, centerFer, v1Fer))
            testNF.append(distance.mahalanobis(test, centerNF, v1NF))
        
    yPred = []
    for i in range(len(xtest)):
        if testFer[i] >= testNF[i]:
            yPred.append(0)
        else:
            yPred.append(1)
                

    f1_sco = f1(ytest, np.array(yPred))
    return f1_sco
        
        

def get_f1_base(xtrain, ytrain, xtest, ytest, numFeat):
    f1Tot = nested(xtrain, ytrain, xtest, ytest, numFeat)
    
    jackTrainArr = []
    jackTestArr = []
            
    for i in range(len(xtrain)):
        x_train = np.delete(np.array(xtrain), i, 0)
        y_train = np.delete(np.array(ytrain), i, 0)
        
        scoreTrain = nested(x_train, y_train, xtest, ytest, numFeat)
        
        jackTrainArr.append(scoreTrain)
            
    for t in range (len(xtest)):
        x_test = np.delete(np.array(xtest), t, 0)
        y_test = np.delete(np.array(ytest), t, 0)
            
        scoreTest = nested(xtrain, ytrain, x_test, y_test, numFeat)
        
        jackTestArr.append(scoreTest)  
            
    return  f1Tot, jackTrainArr, jackTestArr 

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