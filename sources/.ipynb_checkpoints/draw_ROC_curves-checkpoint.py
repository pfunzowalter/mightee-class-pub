import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# do not touch
def threshold (X, Y, direction):
    acc = []
    tp = []     
    fp = []
    
    th = np.array(X).flatten()
    
    inTrain, outTrain = (list(t) for t in zip(*sorted(zip(th, np.array(y).flatten()))))
    
    thresholds = np.arange(inTrain[0], inTrain[-1]+1 , 0.001)
    print(inTrain[0])
    print(inTrain[-1]+1)
    
    for i in thresholds:
        pred = []

        for xTr in inTrain:
            if direction:
                if i > xTr:
                    pred.append(1)
                else:
                    pred.append(0)
            else:
                if i < xTr:
                    pred.append(1)
                else:
                    pred.append(0)

#         print(pred)        
        acc.append(accuracy_score(outTrain, pred))
        CM = confusion_matrix(outTrain, pred)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP) 
        tp.append(TPR)
        fp.append(1-TNR)
        
#         acc.append(accuracy_score(outTest, predTest))        
#         thre.append(i)
        
        
    return acc, fp, tp


def draw_ROC (di, X_train, features, y1):
    lw = 2
    plt.figure(figsize=(10 , 8))
    for f, d in zip(features, di):
        x = np.array(X_train[[f]]).flatten()
        y = np.array(y1).flatten()
        acc, tp, fp = threshold (x, y, d)
        print()
        plt.plot(tp, fp, lw=lw, label = f+" AUC: "+str(round(metrics.auc(tp, fp),2)) )
        
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.title('$MeOH$', fontweight ='bold', fontsize =12)
    plt.xlabel('False Positive Rate', fontweight ='bold', fontsize =14)
    plt.ylabel('True Positive Rate', fontweight ='bold', fontsize =14)
    plt.title('ROC curves based on features', fontweight ='bold', fontsize =14)
    plt.legend(loc="lower right", prop={"size":12})
    plt.show() 