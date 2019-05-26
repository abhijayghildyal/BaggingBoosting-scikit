#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:35:01 2018

@author: abhijay
"""

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import tree

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)

#import os
#os.chdir('/home/abhijay/Documents/ML/hw_3/Q_6/')

##### For making the target variable binary #####
def make_target_variable(data):
    data['salary-bracket'] = data['salary-bracket'].apply(lambda y: 0 if y==" <=50K" else 1)
    return data

#### Sepatate categorical values from continuous for label encoding of the categorical data #####
def find_categorical_continuous_features(data):
    categorical_features = [data.columns[col] for col, col_type in enumerate(data.dtypes) if col_type == np.dtype('O') ]
    continuous_features = list(set(data.columns) - set(categorical_features))
    return categorical_features, continuous_features

def plot_results(train_accuracy, dev_accuracy, test_accuracy, x_label, experiment_name, saveAs):
    w = 0.27
    ind = np.arange(len(train_accuracy))
    fig, ax = plt.subplots(figsize=(20, 8))
    dat = np.array([train_accuracy,dev_accuracy,test_accuracy]).T
    ax.bar(ind, dat[:,0], width=w, label = 'Train Accuracy')
    ax.bar(ind+w, dat[:,1], width=w, label = 'Dev Accuracy')
    ax.bar(ind+w*2, dat[:,2], width=w, label = 'Test Accuracy')
    ax.set_ylabel( 'Accuracy', fontsize=15)
    ax.set_xlabel( x_label, fontsize=15)
    ax.set_xticks(ind+w)
    ax.set_xticklabels(experiment_name)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.set_title(saveAs)
    plt.xticks(rotation=70)
    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig('plots/'+saveAs)

if __name__ == "__main__":

    print ("\n=============== Ans 6 ===============\n")
    
    col_names = ["age","workclass","education","marital-status","occupation","race","gender","hours-per-week","native-country","salary-bracket"]
    
    ##### Load data #####
    train_data = pd.read_csv("income-data/income.train.txt", names = col_names)
    dev_data = pd.read_csv("income-data/income.dev.txt", names = col_names)
    test_data = pd.read_csv("income-data/income.test.txt", names = col_names)
    
    train_data = make_target_variable(train_data)
    test_data = make_target_variable(test_data)
    dev_data = make_target_variable(dev_data)
        
    categorical_features_, continuous_features_ = find_categorical_continuous_features(train_data.iloc[:,0:-1])
    
    categorical_features = [train_data.columns.get_loc(c) for c in categorical_features_]
    
    continuous_features = [train_data.columns.get_loc(c) for c in continuous_features_]
    
    ##### Encoding categorical values to labels #####
    le = preprocessing.LabelEncoder()
    all_df = pd.concat([train_data,test_data,dev_data])
    for feature in categorical_features_:
        le.fit(all_df[feature])
        train_data[feature] = le.transform(train_data[feature])
        test_data[feature] = le.transform(test_data[feature])
        dev_data[feature] = le.transform(dev_data[feature])
    
    featuresUniqueValues = [train_data[col].unique() for col in col_names]
    
    ##### Convert pandas dataframe to numpy array #####
    x = train_data.iloc[:,0:train_data.shape[1]-1].values
    y = (train_data.values)[:,-1]
    
    x_test = test_data.iloc[:,0:test_data.shape[1]-1].values
    y_test = (test_data.values)[:,-1]
    
    x_dev = dev_data.iloc[:,0:dev_data.shape[1]-1].values
    y_dev = (dev_data.values)[:,-1]
    
    ##### Bagging #####
    print ("\n=============== Bagging ===============\n")
    train_accuracy = []
    dev_accuracy = []
    test_accuracy = []
    experiment_name = []
        
    for maxDepthOfTree in [1,2,3,5,10]:
        for noOfTrees in [10, 20, 40, 60, 80, 100]:
            print ("\n\n---------------")
            print ("maxDepthOfTree: ",maxDepthOfTree)
            print ("noOfTrees: ",noOfTrees)
            print ("---------------")
            clf = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=maxDepthOfTree),n_estimators=noOfTrees)
            
            clf = clf.fit(x, y)
            
            train_accuracy_ = round(100*np.sum(y == clf.predict(x))/x.shape[0],2)
            dev_accuracy_ = round(100*np.sum(y_dev == clf.predict(x_dev))/x_dev.shape[0],2)
            test_accuracy_ = round(100*np.sum(y_test == clf.predict(x_test))/x_test.shape[0],2)
            
            train_accuracy.append(train_accuracy_)
            dev_accuracy.append(dev_accuracy_)
            test_accuracy.append(test_accuracy_)
            experiment_name.append("("+str(maxDepthOfTree)+','+str(noOfTrees)+")")
    
            print ("\nTraining Accuracy: "+str( train_accuracy_)+"%")
            print ("\nDev Accuracy: "+str( dev_accuracy_)+"%")
            print ("\nTesting Accuracy: "+str( test_accuracy_)+"%")
    
    saveAs = 'Bagging - Accuracy Plot'
    xlabel = '(Max depth of tree, No. of boosting iterations)' ##### No. of boosting iterations is the same as n_estimators or number of trees
    plot_results(train_accuracy, dev_accuracy, test_accuracy, xlabel, experiment_name, saveAs)
    
    ##### Boosting #####
    print ("\n=============== Boosting ===============\n")
    train_accuracy = []
    dev_accuracy = []
    test_accuracy = []
    experiment_name = []

    for maxDepthOfTree in [1,2,3]:
        for noOfTrees in [10, 20, 40, 60, 80, 100]:
            ##### Boosting iteration is the same as 

            print ("\n\n---------------")
            print ("maxDepthOfTree: ",maxDepthOfTree)
            print ("noOfTrees: ",noOfTrees)
            print ("---------------")
            clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=maxDepthOfTree),n_estimators=noOfTrees)
            
            clf = clf.fit(x, y)
    
            train_accuracy_ = round(100*np.sum(y == clf.predict(x))/x.shape[0],2)
            dev_accuracy_ = round(100*np.sum(y_dev == clf.predict(x_dev))/x_dev.shape[0],2)
            test_accuracy_ = round(100*np.sum(y_test == clf.predict(x_test))/x_test.shape[0],2)
            
            train_accuracy.append(train_accuracy_)
            dev_accuracy.append(dev_accuracy_)
            test_accuracy.append(test_accuracy_)
            experiment_name.append("("+str(maxDepthOfTree)+','+str(noOfTrees)+")")
    
            print ("\nTraining Accuracy: "+str( train_accuracy_)+"%")
            print ("\nDev Accuracy: "+str( dev_accuracy_)+"%")
            print ("\nTesting Accuracy: "+str( test_accuracy_)+"%")
            
    saveAs = 'Boosting - Accuracy Plot'
    xlabel = '(Max depth of tree, No. of trees)'
    plot_results(train_accuracy, dev_accuracy, test_accuracy, xlabel, experiment_name, saveAs)