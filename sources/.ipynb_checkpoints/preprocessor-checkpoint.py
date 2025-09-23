import missingno as msno
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl

def data_processor( data, x_features, y_features, binary_classification = True ):
    """ This function prepares the samples a subset of features for machine learning from a high dimensional data.
        classification : type of classification, binary == SFG or AGNs, Other wise classification will be for AGNs
        or SFG or noclass
        data = high dimensional dataset (MIGHTEE data)
        X-features = columns of interest contain data
        Y-features = the output features 
        
    """
    if binary_classification == True:
        # extracting the x-features
        x_sets = []
        for i in x_features:
            x = data[i]
            x_sets.append(np.array(x))
    
        X = np.vstack((x_sets)).T
    
        #extracting the y-feature
        y_sets = []
        for l in y_features:
            y = data[l]
            y_sets.append(y)
        
        y = np.vstack((y_sets)).T
    
        # converting the features into the data frame
        y_data = pd.DataFrame(y, columns = y_features)
        X_data = pd.DataFrame(X, columns = x_features)

        # joinin the two data sets into one dataframe
        mightee_data = pd.concat([X_data, y_data], axis=1, sort=False)
    
        # Sampling the sources that are classified as midIRAGB = AGN and the sources that are classified as notmidIRAGN = SFG
        AGN = mightee_data[mightee_data[y_features[0]] == True]
        SFG = mightee_data[mightee_data[y_features[1]] == True]
        probSFG = mightee_data[mightee_data[y_features[2]] == True]
        unclass = mightee_data[mightee_data[y_features[3]] == True]
        
        print('shape of the AGN: ', AGN.shape)
        print('shape of the SFG: ', SFG.shape)
        print('shape of the probSFG: ', probSFG.shape)
        print('shape of the unclass: ', unclass.shape)
        print('total sample: ', len(AGN) + len(SFG)+len(probSFG)+len(unclass))
    
        # We now drop the not column
        mightee_agn = AGN.drop([y_features[1], y_features[2]], axis = 1)
        mightee_sfg = SFG.drop([y_features[1], y_features[2]], axis = 1)
        mightee_probsfg = probSFG.drop([y_features[1], y_features[2]], axis = 1)

        # We now replace True with the true label AGN or SFG for the corresponding source
        mightee_agn1 = mightee_agn.replace(True, 'AGN', regex=True)
        mightee_sfg1 = mightee_sfg.replace(False, 'SFG', regex=True)
        mightee_probsfg1 = mightee_probsfg.replace(False, 'SFG', regex=True)
    
        # combining this data into one
        complete_mightee = pd.concat([mightee_agn1, mightee_sfg1, mightee_probsfg1], sort=False)
        complete_mightee1 = complete_mightee.replace(-np.inf, np.nan, regex=True) 
        catalogue = complete_mightee1.drop("unclass", axis='columns')
        print("DONE PROCESSING")
    
        return catalogue #, unclass 





    else:
        x_sets = []
        for i in x_features:
            x = data[i]
            x_sets.append(np.array(x))

        X = np.vstack((x_sets)).T

        #extracting the y-feature
        y_sets = []
        for l in y_features:
            y = data[l]
            y_sets.append(y)

        y = np.vstack((y_sets)).T

        # converting the features into the data frame
        y_data = pd.DataFrame(y, columns = y_features)
        X_data = pd.DataFrame(X, columns = x_features)

        # joinin the two data sets into one dataframe
        mightee_data = pd.concat([X_data, y_data], axis=1, sort=False)

        # Sampling the sources that are classified as midIRAGB = AGN and the sources that are classified as notmidIRAGN = SFG
        AGN = mightee_data[mightee_data[y_features[0]] == True]
        SFG = mightee_data[mightee_data[y_features[1]] == True]
        probSFG = mightee_data[mightee_data[y_features[2]] == True]
        noclass = mightee_data[(mightee_data[y_features[0]] == False) & (mightee_data[y_features[1]] == False) & (mightee_data[y_features[2]] == False)]

        print('shape of the AGN: ', AGN.shape)
        print('shape of the SFG: ', SFG.shape)
        print('shape of the probSFG: ', probSFG.shape)
        print('shape of unclassified: ',noclass.shape)
        print('total sample: ', len(AGN) + len(SFG) + len(probSFG) + len(noclass))

        # We now drop the not column
        mightee_agn = AGN.drop([y_features[1], y_features[2]], axis = 1)
        mightee_sfg = SFG.drop([y_features[1], y_features[2]], axis = 1)
        mightee_probsfg = probSFG.drop([y_features[1], y_features[2]], axis = 1)
        mightee_noclass = noclass.drop([y_features[1], y_features[2]], axis = 1)

        # We now replace True with the true label AGN or SFG for the corresponding source
        mightee_agn1 = mightee_agn.replace(True, 'AGN', regex=True)
        mightee_sfg1 = mightee_sfg.replace(False, 'SFG', regex=True)
        mightee_probsfg1 = mightee_probsfg.replace(False, 'SFG', regex=True)
        mightee_noclass1 = mightee_noclass.replace(False, 'noclass', regex=True)

        # combining this data into one
        complete_mightee = pd.concat([mightee_agn1, mightee_sfg1, mightee_probsfg1, mightee_noclass1], sort=False)
        complete_mightee1 = complete_mightee.replace(-np.inf, np.nan, regex=True) 
        print("DONE PROCESSING")
    
        return complete_mightee1