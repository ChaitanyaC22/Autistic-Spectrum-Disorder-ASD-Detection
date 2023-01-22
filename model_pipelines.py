# -*- coding: utf-8 -*-
# Custom Module 2: 'model_pipelines.py'
'''
Contains all the model pipelines to build models for 
this Autism spectrum disorder (ASD) status-check binary classification task
'''
##Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Necessary Libraries
# Importing all the necessary libraries
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)       # To display all columns in pandas dataframe
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.io.arff import loadarff as load_arff
from pprint import pprint
import statistics
from statistics import mode
import collections
from collections import Counter
import sklearn
import random
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
## xgboost packages
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
random.seed(0)
np.random.seed(0)

import helper_functions
from helper_functions import *


class classifier_pipelines:
    
    # Importing the 'helper_class' from 'helper_functions' module
    def __init__(self):
        self.magic_helper_func = magic_helper()
    

    def LogisticRegression_KFoldCV_Model(self, X_train, y_train, X_test, y_test, model_name, results_df):
        """
        Builds optimal logistic regression estimator by cross-validating, tuning hyperparameters 
        and evaluates the model

        Parameters:
        X_train : Input features from the train dataset
        y_train : True labels from the train dataset
        X_test : Input features from the test dataset
        y_test : True labels from the test dataset (For evaluation)
        model_name (str): Desired name of the model/estimator
        results_df (pandas.DataFrame): Dataframe containing model metric scores

        Returns:
        results_df (pandas.DataFrame): Updated results dataframe containing model-wise metric scores
        clf_opt: optimal logistic regression estimator obtained after hyperparameter tuning and K-Fold cross-validation
        """
        start= time.time();print('\033[1m'+"*"*100+'\033[0m')
        np.random.seed(0)

        # Performing cross-validation and hyperparameter tuning
        num_C= list(np.power(10.0, np.arange(-4,5)))  # num_C (i.e. Cs values) (Hyperparameter of LogisticRegression Model)

        # Create a pipeline: 
        pipe_lr= make_pipeline(LogisticRegression(class_weight='balanced',     # Auto-frequency based class-weights
                                                  n_jobs=-1,max_iter=1000,random_state=0))  # Define Logistic Regression Model 

        # Param_distributions (Create both estimators: find optimal estimator- L1/L2 regularization)
        params_lr= [{ 
                    "logisticregression__penalty": ['l2'],
                    "logisticregression__C":num_C,
                    "logisticregression__solver":['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear'] # solvers that allow L2 penalty
                    },
                    {
                    "logisticregression__penalty": ['l1'],
                    "logisticregression__C":num_C,
                    "logisticregression__solver":['saga', 'liblinear'] # solvers that allow L1 penalty
                    }]
        clf= GridSearchCV(                                             # Performs Cross-validation 
                      estimator=pipe_lr, 
                      param_grid=params_lr,                            # Hyperparameters to be tuned
                      n_jobs=-1, 
                      cv=5,                                            # KFold Cross validation (5-Fold CV)
                      scoring='accuracy',                              # Scoring metric 'accuracy'
                      verbose = 1,
                      return_train_score=True,
                      error_score=0)
        
        clf.fit(X_train,y_train)
        optimal_solver= clf.best_params_['logisticregression__solver']        ## Storing optimal hyperparameters in variables
        optimal_penalty= clf.best_params_['logisticregression__penalty']
        optimal_C= float(clf.best_params_['logisticregression__C'])
        self.magic_helper_func.best_cross_val_results(clf, model_name)        ## Get best cross_validation result
        
        ##Initialize the classifier with optimal hyperparameters  
        clf_opt= clf.best_estimator_._final_estimator    ## Best estimator already found out above, after fitting on (X_train,y_train)
        
        print("Optimal hyperparameters:")
        print("Best C value:", optimal_C)
        print("Best penalty:", optimal_penalty)
        print("Best solver:", optimal_solver)
        # Prediction and evaluation results on actual TEST data
        results_store_df= self.magic_helper_func.prediction_evaluation_results(clf, X_train, y_train, 
                                                        X_test, y_test,
                                                        model_name, results_df) 
        results_df= results_df.append(results_store_df, ignore_index=True)   ## Appending the results to 'results_df' dataframe
        end= time.time()
        time_req_secs = (end-start)
        if time_req_secs>=60:
            time_req_mins= time_req_secs/60
            print(f"\nTime required to train the model: {round(time_req_mins)} minutes")
        else:
            print(f"\nTime required to train the model: {round(time_req_secs)} seconds")
        print('\033[1m'+"*"*100+'\033[0m')
        return results_df, clf_opt
    
    

    def RandomForestClassifier_KFoldCV_Model(self, X_train, y_train, X_test, y_test, model_name, results_df):
        """
        Builds optimal random forest classifier/estimator by cross-validating, tuning hyperparameters 
        and evaluates the model

        Parameters:
        X_train : Input features from the train dataset
        y_train : True labels from the train dataset
        X_test : Input features from the test dataset
        y_test : True labels from the test dataset (For evaluation)
        model_name (str): Desired name of the model/estimator
        results_df (pandas.DataFrame): Dataframe containing model metric scores

        Returns:
        results_df (pandas.DataFrame): Updated results dataframe containing model-wise metric scores
        clf_opt: optimal random forest classifier/estimator obtained after hyperparameter tuning and K-Fold cross-validation
        """
        start= time.time()
        print('\033[1m'+"*"*100+'\033[0m')
        np.random.seed(0)

        ## Perform cross-validation and hyperparameter tuning 
        # Create a pipeline: 
        pipe_rfc= make_pipeline(RandomForestClassifier(warm_start=True,                          # Defining the model
                                                       verbose=0,
                                                       n_jobs=-1, random_state=0))                

        # Hyperparameters grid
        n_estimators=[100,300,500,600,850,1000,1200]              ## Hyperparameter: n_estimators to be tuned
        min_samples_split=[2,5,7,10]                                  ## Hyperparameter: min_samples_split to be tuned
        min_samples_leaf= [1,2,4]                                     ## Hyperparameter: min_samples_leaf to be tuned
        max_features=['auto', 'sqrt', 'log2', None]                   ## Hyperparameter: max_features to be tuned        
        max_depth= [3,5,7,9]                                          ## Hyperparameter: max_depth to be tuned
        criterion= ['gini','entropy']                                 ## Hyperparameter: criterion to be tuned

        ## Param_distributions
        params_rfc={
                    'randomforestclassifier__n_estimators': n_estimators,
                    'randomforestclassifier__min_samples_split':min_samples_split,
                    'randomforestclassifier__min_samples_leaf':min_samples_leaf,
                    'randomforestclassifier__max_features': max_features,
                    'randomforestclassifier__max_depth':max_depth,
                    'randomforestclassifier__criterion': criterion}

        clf= RandomizedSearchCV(                                              ## Performing cross-validation
                        estimator=pipe_rfc, 
                        param_distributions=params_rfc, 
                        n_jobs=-1, 
                        cv=5,                                          # KFold Cross validation (5-Fold CV)
                        n_iter=100,                                    # Number of parameter settings ran
                        scoring='accuracy',                            # Scoring metric 'accuracy'
                        verbose = 1,
                        return_train_score=True,
                        error_score=0)

        clf.fit(X_train, y_train) 
        optimal_n_estimators= int(clf.best_params_['randomforestclassifier__n_estimators'])
        optimal_min_samples_split= int(clf.best_params_['randomforestclassifier__min_samples_split'])
        optimal_min_samples_leaf= int(clf.best_params_['randomforestclassifier__min_samples_leaf'])
        optimal_max_features= clf.best_params_['randomforestclassifier__max_features']
        optimal_max_depth= int(clf.best_params_['randomforestclassifier__max_depth'])
        optimal_criterion= clf.best_params_['randomforestclassifier__criterion']
        self.magic_helper_func.best_cross_val_results(clf, model_name)      # Get best cross_validation results
        ##Initialize the classifier with optimal hyperparameters
        clf_opt = clf.best_estimator_._final_estimator      ## Best estimators found earlier, already fit on (X_train, y_train)              

        print("Optimal hyperparameters:")
        print("Best number of trees:", optimal_n_estimators)
        print("Best min_samples_split:", optimal_min_samples_split)
        print("Best min_samples_leaf:", optimal_min_samples_leaf)
        print("Best max_features:", optimal_max_features)
        print("Best max_depth:", optimal_max_depth)
        print("Best criterion:", optimal_criterion)
        # prediction and evaluation results on actual TEST data
        results_store_df= self.magic_helper_func.prediction_evaluation_results(clf, X_train, y_train, 
                                                        X_test, y_test, 
                                                        model_name, results_df) 
        results_df= results_df.append(results_store_df, ignore_index=True)   ## Appending the results to 'results_df' dataframe
        end= time.time()
        time_req_secs = (end-start)
        if time_req_secs>=60:
            time_req_mins= time_req_secs/60
            print(f"\nTime required to train the model: {round(time_req_mins)} minutes")
        else:
            print(f"\nTime required to train the model: {round(time_req_secs)} seconds")
        print('\033[1m'+"*"*100+'\033[0m')
        return results_df, clf_opt
    



    def XGBoostClassifier_KFoldCV_Model(self, X_train, y_train, X_test, y_test, model_name, results_df):
        """
        Builds optimal XGBoost classifier/estimator by cross-validating, tuning hyperparameters 
        and evaluates the model

        Parameters:
        X_train : Input features from the train dataset
        y_train : True labels from the train dataset
        X_test : Input features from the test dataset
        y_test : True labels from the test dataset (For evaluation)
        model_name (str): Desired name of the model/estimator
        results_df (pandas.DataFrame): Dataframe containing model metric scores

        Returns:
        results_df (pandas.DataFrame): Updated results dataframe containing model-wise metric scores
        clf_opt: optimal XGBoost classifier/estimator obtained after hyperparameter tuning and K-Fold cross-validation
        """
        start= time.time()
        print('\033[1m'+"*"*100+'\033[0m')

        np.random.seed(0)
        ## Perform cross-validation and hyperparameter tuning 
        # Create a pipeline: 
        pipe_xgb= make_pipeline(XGBClassifier(objective='binary:logistic',    # Defining the model (For binary classification)
                                                  n_jobs=-1,verbosity=0, random_state=0))

        # Define list of hyperparameters for tuning
        learning_rate= [0.05, 0.1, 0.2, 0.3]                            # Hyperparameter: (learning_rate) to be tuned  
        n_estimators= [10,50,100,200,300,500,700,900,1000,1200]         # Hyperparameter: n_estimators to be tuned
        max_depth= list(range(3,10,2))                                  # Hyperparameter: 'max_depth' to be tuned

        # Param_distributions
        params_xgb={
                    'xgbclassifier__n_estimators':n_estimators,
                    'xgbclassifier__max_depth': max_depth,
                    'xgbclassifier__learning_rate':learning_rate}   

        clf= GridSearchCV(                                             ## Performs Cross-validation 
                                estimator=pipe_xgb, 
                                param_grid=params_xgb, 
                                n_jobs=-1, 
                                cv=5,                                   # KFold Cross validation (5-Fold CV)
                                scoring='accuracy',                     # Scoring metric 'accuracy'
                                verbose=1,
                                return_train_score=True,
                                error_score=0)

        clf.fit(X_train, y_train)
        optimal_n_estimators= int(clf.best_params_['xgbclassifier__n_estimators'])
        optimal_max_depth= int(clf.best_params_['xgbclassifier__max_depth'])
        optimal_learning_rate= float(clf.best_params_['xgbclassifier__learning_rate'])
        self.magic_helper_func.best_cross_val_results(clf, model_name)     ## Get best cross_validation results
        ##Initialize the classifier with optimal hyperparameters
        clf_opt = clf.best_estimator_._final_estimator      ## Best estimators found earlier, already fit on (X_train, y_train)              

        print("Optimal hyperparameters:")
        print("Best number of trees:", optimal_n_estimators)
        print("Best max depth:", optimal_max_depth)
        print("Best learning rate:", optimal_learning_rate)
        # prediction and evaluation results on actual TEST data
        results_store_df= self.magic_helper_func.prediction_evaluation_results(clf, X_train, y_train, X_test, y_test,
                                                    model_name, results_df) 
        results_df= results_df.append(results_store_df, ignore_index=True)   ## Appending the results to 'results_df' dataframe
        end= time.time()
        time_req_secs = (end-start)
        if time_req_secs>=60:
            time_req_mins= time_req_secs/60
            print(f"\nTime required to train the model: {round(time_req_mins)} minutes")
        else:
            print(f"\nTime required to train the model: {round(time_req_secs)} seconds")
        print('\033[1m'+"*"*100+'\033[0m')
        return results_df, clf_opt
    
    


    def KNeighborsClassifier_KFoldCV_Model(self, X_train, y_train, X_test, y_test, model_name, results_df):
        """
        Builds optimal KNeighbors classifier/estimator by cross-validating, tuning hyperparameters 
        and evaluates the model

        Parameters:
        X_train : Input features from the train dataset
        y_train : True labels from the train dataset
        X_test : Input features from the test dataset
        y_test : True labels from the test dataset (For evaluation)
        model_name (str): Desired name of the model/estimator
        results_df (pandas.DataFrame): Dataframe containing model metric scores

        Returns:
        results_df (pandas.DataFrame): Updated results dataframe containing model-wise metric scores
        clf_opt: optimal KNeighbors classifier/estimator obtained after hyperparameter tuning and K-Fold cross-validation
        """
        start= time.time()
        print('\033[1m'+"*"*100+'\033[0m') 

        np.random.seed(0)

        # Hyperparameter 'k' value list
        k_range = list(range(3,50,1))                                    ## Iterating over range of odd K values from 3 to 49

        # Create a pipeline:
        pipe_knn= make_pipeline(KNeighborsClassifier(n_jobs=-1))                # Defining the model 

        # Param_distributions
        params_knn= {'kneighborsclassifier__n_neighbors': k_range}                   ## List of hyperparameters to be tuned                            
        clf= GridSearchCV(                                                     ## Performs Cross-validation 
                                estimator=pipe_knn, 
                                param_grid=params_knn, 
                                n_jobs=-1, 
                                cv=5,                                   # KFold Cross validation (5-Fold CV)
                                scoring='accuracy',                     # Scoring metric 'accuracy'
                                verbose = 1,
                                return_train_score=True,
                                error_score=0)

        clf.fit(X_train, y_train)
        optimal_n_neighbors= int(clf.best_params_['kneighborsclassifier__n_neighbors'])
        self.magic_helper_func.best_cross_val_results(clf, model_name)
        ##Initialize the classifier with optimal hyperparameters
        clf_opt= clf.best_estimator_._final_estimator    ## Best estimator found earlier and already fit on (X_train,y_train)                             

        print("Optimal hyperparameters:")
        print("Best n_neighbors (K):", optimal_n_neighbors)
        # prediction and evaluation results on actual TEST data
        results_store_df= self.magic_helper_func.prediction_evaluation_results(clf, X_train, y_train, 
                                                        X_test, y_test,
                                                        model_name, results_df) 
        results_df= results_df.append(results_store_df, ignore_index=True)   ## Appending the results to 'results_df' dataframe
        end= time.time()
        time_req_secs = (end-start)
        if time_req_secs>=60:
            time_req_mins= time_req_secs/60
            print(f"\nTime required to train the model: {round(time_req_mins)} minutes")
        else:
            print(f"\nTime required to train the model: {round(time_req_secs)} seconds")
        print('\033[1m'+"*"*100+'\033[0m')
        return results_df, clf_opt