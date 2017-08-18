#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sumedha Singh
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from sklearn import preprocessing
from scipy.stats import skew

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error

#user input train/test file path
train_path = sys.argv[1]
test_path = sys.argv[2]

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

target_col = ["groundtruth_walkin"]



def plotSkewedDistribution( file, attrName ):
    attrTransformed = preprocessing.scale(np.sqrt(file[attrName])) 
    attrOrig = preprocessing.scale(file[attrName])
    
    #skness = skew(AirTime)
    
    skness = skew(attrTransformed)
    sknessOrig = skew(attrOrig)
    
    figure = plt.figure()
    figure.add_subplot(131)   
    plt.hist(attrTransformed,facecolor='red',alpha=0.75) 
    plt.xlabel("Transformed(Using Sqrt)") 
    plt.title("Transformed Histogram") 
    plt.text(3,110,"Skewness: {0:.2f}".format(skness)) 
    
    figure.add_subplot(132) 
    plt.hist(attrOrig,facecolor='blue',alpha=0.75) 
    plt.xlabel("Based on Original video_walkin values") 
    plt.title("Histogram - Right Skewed") 
    plt.text(4,105,"Skewness: {0:.2f}".format(sknessOrig))
    figure.add_subplot(133) 
    plt.boxplot(attrTransformed)
    plt.title("Un-Skewed Distribution")
    plt.show()
    
    #setting the size of the figure
    figure.set_size_inches(18.5, 10.5)
    #save the plot as a figure
    figure.savefig('SkewedDistributionPlot.png', dpi=100)
    
def transformSkewness(data):
    #print skewness for each attribute
    print("Skewness for each feature:\n")
    for attr in data.columns:
        print(attr,' : ',skew(preprocessing.scale(data[attr])))
        
    #list of attributes that are highly skewed
    pos_skew_attrs = ['video_walkin', 'video_walkout', 'wifi_walkin', 'wifi_walkout', 'sales_in_next_15_to_30_min','predict_walkin', 'predict_walkout'] 

    #Transform the file dataframe to remove skewness
    for attr in pos_skew_attrs:
        data[attr] = preprocessing.scale(np.sqrt(data[attr]))
    
    return data
    
def selectFeatures(file):
    # Random forest Extra tree model for selecting top features
    model = ExtraTreesClassifier()
    model.fit(file, file[list(target_col)])
    # Print Relative Feature Importance
    print(model.feature_importances_)
    
    #feature selection using chi Square statistics
    array = file.values
    X = array[:,0:13]
    Y = array[:,13]
    test = SelectKBest(score_func=chi2, k=12)
    fit = test.fit(X, Y)
    # summarize scores
    np.set_printoptions(precision=3)
    features = fit.transform(X)
    imp_features_chi2 = features
    print(imp_features_chi2)


def trainModels(file, imp_attr):
    #split into train and test
    train, test =  train_test_split(file, test_size = 0.2)

    #Removing the target/predictor from the train data
    targ = train[list(target_col)]
    train.drop('groundtruth_walkin', axis=1, inplace=True)
    
    ### Random Forest Model
    rand_forest_model =  RandomForestRegressor(n_estimators = 1000 , max_features = 2, oob_score = True ,  random_state = 115)
    rand_forest_model.fit(train[list(imp_attr)],targ)
    
    ### Decision Tree Model
    decision_tree_model = DecisionTreeRegressor(max_depth=4)
    decision_tree_model.fit(train[list(imp_attr)],targ)    
   
    ### Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(train[list(imp_attr)], targ)
    
    return rand_forest_model, decision_tree_model, linear_model, test

def  testNEvalModels(test, rf_model, dt_model, lm_model, imp_attr):
   
    print('\n Evaluation Staistics:')
    ### Evaluating Random Forest
    print("\n ***Random Forest Regressor***")
    #Evaluation metric: r square
    r2 = r2_score(test[list(target_col)] , rf_model.predict(test[list(imp_attr)]))
    print("R-Square Value:", r2)
    
    # extracting the test target values and convert to float
    true_vals = test[list(target_col)].values
    true_vals_flt = true_vals.astype(np.float)
    
    prediction = rf_model.predict(test[list(imp_attr)])
    
    #reshaping the array is required to convert it into numpy array
    aa = prediction.reshape(-1,1)
    
    mean_squared_error(true_vals_flt, prediction)
    
    mse = np.mean((true_vals_flt - aa)**2)
    print("Mean Squared Error", mse)
    
    print("Explained Variance Score", explained_variance_score(true_vals_flt, prediction))     
    print("Mean Absolute Error", mean_absolute_error(true_vals_flt, prediction))
    print("Median Absolute Error", median_absolute_error(true_vals_flt, prediction))    
    
    ### Evaluating Decision tree
    print("\n ***Decision Tree Regressor***")
    y_2 = dt_model.predict(test[list(imp_attr)])
    r2_dt = r2_score(true_vals_flt , y_2)
    print("R-Square Value:", r2_dt)
    
    print("Mean Squared Error", mean_squared_error(true_vals_flt, y_2))  
    
    print("Explained Variance Score", explained_variance_score(true_vals_flt, y_2))     
    print("Mean Absolute Error", mean_absolute_error(true_vals_flt, y_2))
    print("Median Absolute Error", median_absolute_error(true_vals_flt, y_2))    
    
    
    ### Evaluating Linear Model
    print("\n ***Linear Regression Model***")
    pred_lm = lm_model.predict(test[list(imp_attr)])

    r2_lm = r2_score(true_vals_flt, pred_lm)
    
    print("R-Square Value:", r2_lm)
    
    print("Mean Squared Error", mean_squared_error(true_vals_flt, pred_lm))  
    
    print("Explained Variance Score", explained_variance_score(true_vals_flt, pred_lm))     
    print("Mean Absolute Error", mean_absolute_error(true_vals_flt, pred_lm))
    print("Median Absolute Error", median_absolute_error(true_vals_flt, pred_lm))    

    
def applyModel(model, test):
    #Apply selected model to test data
    pred = model.predict(test[list(imp_attr)])
    
    #convert prediction to nearest integer
    pred_int = np.rint(pred)
    
    #save predicted into target column in test
    test_data['groundtruth_walkin'] = pred_int
    
    #save df to file
    test_data.to_csv(test_path)
    
    
#Data processing and model building
plotSkewedDistribution(train_data, attrName = "video_walkin")   
selectFeatures(train_data)

#features selected based on the result of selectFeatures function
imp_attr = ['video_walkin', 'video_walkout','device_angle', 'average_person_size', 'distance_to_door', 'mall_or_street',
       'predict_walkin', 'predict_walkout', 'wifi_walkin', 'wifi_walkout', 'sales_in_next_15_to_30_min']

#Resolve skewness using the Square root transform
train_data = transformSkewness(train_data)

rf,dt,lm,test = trainModels(train_data, imp_attr)
testNEvalModels(test, rf, dt, lm, imp_attr)

# As per the Evaluation Statistics we get to know that Random Forest 
# is performing better than other considered models, so we apply rf to test data
applyModel(rf, test_data)

 