# -*- coding: utf-8 -*-
# Custom Module 1: 'helper_functions.py'
'''
Contains all the necessary helper functions or methods that are needed
 to plot graphs, vizualize data and other miscellaneous related operations
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
random.seed(0)
np.random.seed(0)

# Install the following package/s, if they are unavailable in the environment
# !pip install statistics
# import inspect 
# my_packages_path = os.path.dirname(inspect.getfile(inspect))+"/site-packages"
# !python -m pip install pycountry-convert -t my_packages_path
import pycountry_convert as pc




# Class for helper functions
class magic_helper:

    @staticmethod
    def unique_vals(df, column_list):
        """
        Extracts unique values and their count from a dataset for a list of categorical variables.

        Parameters:
        df (pandas.DataFrame): The dataset to extract unique values from
        column_list (List[str]): List of categorical (object type) variables to extract unique values from

        Returns:
        dict: A dictionary containing the set of unique values and their count for each variable in `column_list`
        """
        df1 = df.copy()
        for i in column_list:
            print(f"Unique values of '{i}' variable:' {set(df1[i])}\nNumber of unique items in '{i}':'{len(set(df1[i]))}\n")

    @staticmethod
    def country_to_continent(country_name):
        """
        Given a country name, returns the continent name it belongs to.

        Parameters:
        country_name (str): The name of the country to find the continent for

        Returns:
        str: The continent name that the country belongs to.
        """
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
    

    @staticmethod
    def create_bar_plot_target(df):
        """
        Create a dual barplot to visualize the distribution of a target variable 'Class/ASD'

        Parameters:
        df (pandas.DataFrame): The dataframe to create the barplots from
        target_col (str): The name of the target variable 'Class/ASD' to visualize

        Outputs:
        Two barplots:
        1. Count vs Target variable 'Class/ASD'
        2. %Count vs Target variable 'Class/ASD'
        """
        classes=df['Class/ASD'].value_counts()
        non_ASD_share=classes[0]/df['Class/ASD'].count()*100
        ASD_share=classes[1]/df['Class/ASD'].count()*100

        ## Creating a dataframe containing number and percentage of classes
        class_data= [["Non-ASD Count", classes[0], non_ASD_share],\
                     ["ASD Count", classes[1], ASD_share]]
        df_target= pd.DataFrame(class_data, columns=["Class/ASD", "count", "count_percentage"])


        plt.figure(figsize=(12,6), dpi=200)
        sns.set_style("whitegrid")


        ##Subplot_121 (Barplot for Number of Non-ASD Count vs ASD Count)
        plt.subplot(1,2,1)
        ax_1= sns.barplot(data = df_target,x= "Class/ASD", y="count", palette=("Paired"))
        ax_1.set_xticklabels(labels=['Non-ASD Count','ASD Count'], fontsize=11)
        plt.title("Non-ASD vs ASD (Count)", fontsize=15, fontweight='bold', y=1.02)
        plt.ylabel("Count", fontsize=14, fontstyle='italic')
        plt.xlabel("Target Variable - Class/ASD", fontsize=14, fontstyle='italic')
        plt.yticks(fontsize=10)

        for i in ax_1.patches:
            ax_1.annotate("{0:.0f}".format(i.get_height()), (i.get_x() + i.get_width() / 2.\
                                                         , i.get_height()), ha = 'center'\
                , va = 'top' , xytext = (0, 10), textcoords = 'offset points',rotation=0, fontsize=12)


        ##Subplot_122 (Barplot for Percentage of Non-ASD Count vs ASD Count)
        plt.subplot(1,2,2)
        ax_2= sns.barplot(data = df_target,x= "Class/ASD", y="count_percentage", palette=("Paired"))
        ax_2.set_xticklabels(labels=['Non-ASD Count','ASD Count'], fontsize=11)
        plt.title("Non-ASD vs ASD Count Percentage(%)", fontsize=15, fontweight='bold', y=1.02)
        plt.ylabel("Count Percentage(%)", fontsize=14, fontstyle='italic')
        plt.xlabel("Target Variable - Class/ASD", fontsize=14, fontstyle='italic')
        plt.yticks(fontsize=9)

        for j in ax_2.patches:
            ax_2.annotate("{0:.2f}".format(j.get_height())+"%", (j.get_x() + j.get_width() / 2.\
                                                         , j.get_height()), ha = 'center'\
                , va = 'top' , xytext = (0, 10), textcoords = 'offset points',rotation=0, fontsize=12)

        plt.tight_layout()
        plt.show()


    
    @staticmethod
    def create_barplots_demographics_1(df):
        """
        Create frequency barplots for the 'Country of residence' and 'Continent' features

        Parameters:
        df (pandas.DataFrame): The dataframe to create the barplots from

        Outputs:
        Two barplots:
        1. 'Country of residence' feature
        2. 'Continent' feature
        """
        sns.set(style='white')
        plt.figure(figsize=(26, 26))

        # Subplot2: Country of residence frequency plot
        plt.subplot(211)
        cn = pd.DataFrame((df["Country_of_res"].value_counts(normalize=True)*100).round(2).sort_values())
        cn.reset_index(inplace=True)
        ax_3 = sns.barplot(x='index',y='Country_of_res', data=cn, palette='RdYlGn')

        plt.setp(ax_3.get_xticklabels(), rotation=60, horizontalalignment='right')
        plt.xlabel('Country of Residence of Test Takers', fontsize=28, fontstyle='italic')
        plt.ylabel('Frequency(%count of total)', fontsize=28, fontstyle='italic')
        plt.title('Country of Residence Analysis', fontsize=30, fontweight='bold', y=1.02)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylim(0,20)
        plt.grid(True)
        for i in ax_3.patches:
            ax_3.annotate(format(i.get_height(), '.2f')+"%", (i.get_x() + i.get_width() / 2.\
                                                            , i.get_height()), ha = 'center'\
                        , va = 'center', rotation=65, xytext = (0, 27), textcoords = 'offset points', fontsize=22)

        ## Subplot2: Continent frequency plot
        plt.subplot(212)
        cn = pd.DataFrame((df["Continent"].value_counts(normalize=True)*100).round(2).sort_values())
        cn.reset_index(inplace=True)
        ax_4 = sns.barplot(x='index',y='Continent', data=cn, palette='Set1')

        plt.setp(ax_4.get_xticklabels(), horizontalalignment='right')
        plt.xlabel("Continent - Based on Test Taker\'s Country of Residence", fontsize= 28, fontstyle='italic')
        plt.ylabel('Frequency(%count of total)', fontsize=28, fontstyle='italic')
        plt.title("Test Taker\'s Demographics - Continent Analysis", fontsize=30, fontweight='bold', y=1.02)
        plt.xticks(fontsize=28, rotation=30)
        plt.yticks(fontsize=28)
        plt.grid(True)
        for i in ax_4.patches:
            ax_4.annotate(format(i.get_height(), '.2f')+"%", (i.get_x() + i.get_width() / 2.\
                                                            , i.get_height()), ha = 'center'\
                        , va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=30)
        plt.tight_layout(pad=2.0)
        plt.show()        

        
    @staticmethod
    def create_barplots_demographics_2(df):
        """
        Create frequency barplots for visualizing test taker's 'Ethnicity', 'Relation', and 'Age_desc' feature data.

        Parameters:
        df (pandas.DataFrame): The dataframe to create the barplots from

        Outputs:
        Three barplots:
        1. 'Ethnicity' feature
        2. 'Relation' feature
        3. 'Age_desc' feature
        """
        sns.set(style='white')
        plt.figure(figsize=(26, 39))

        ## Subplot1: Enthnicity frequency plot
        plt.subplot(311)
        cn = pd.DataFrame((df["Ethnicity"].value_counts(normalize=True)*100).round(2).sort_values())
        cn.reset_index(inplace=True)
        ax_5 = sns.barplot(x='index',y='Ethnicity', data=cn, palette='Set2')

        plt.setp(ax_5.get_xticklabels(), horizontalalignment='right')
        plt.xlabel("Ethnicity - Based on Test Taker\'s Demographics Data", fontsize=28, fontstyle='italic')
        plt.ylabel('Frequency(%count of total)', fontsize= 28, fontstyle='italic')
        plt.title("Test Taker\'s Demographics - Ethnicity Analysis", fontsize=30, fontweight='bold', y=1.02)
        plt.xticks(fontsize=28, rotation=30)
        plt.yticks(fontsize=25)
        plt.grid(True)
        for i in ax_5.patches:
            ax_5.annotate(format(i.get_height(), '.2f')+"%", (i.get_x() + i.get_width() / 2.\
                                                            , i.get_height()), ha = 'center'\
                        , va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=26)


        ## Subplot2: Relation frequency plot
        plt.subplot(312)
        cn = pd.DataFrame((df["Relation"].value_counts(normalize=True)*100).round(2).sort_values())
        cn.reset_index(inplace=True)
        ax_6 = sns.barplot(x='index',y='Relation', data=cn, palette='Set2')

        plt.setp(ax_6.get_xticklabels(), horizontalalignment='right')
        plt.xlabel("Relation - Based on Test Taker\'s Demographics Data", fontsize= 28, fontstyle='italic')
        plt.ylabel('Frequency(%count of total)', fontsize= 28, fontstyle='italic')
        plt.title("Test Taker\'s Demographics - Relation Analysis", fontsize=30, fontweight='bold', y=1.02)
        plt.xticks(fontsize=28, rotation=30)
        plt.yticks(fontsize=25)
        plt.grid(True)
        for i in ax_6.patches:
            ax_6.annotate(format(i.get_height(), '.2f')+"%", (i.get_x() + i.get_width() / 2.\
                                                            , i.get_height()), ha = 'center'\
                        , va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=25)


        # Subplot3: Age Description frequency plot
        plt.subplot(313)
        cn = pd.DataFrame((df["Age_desc"].value_counts(normalize=True)*100).round(2).sort_values())
        cn.reset_index(inplace=True)
        ax_7 = sns.barplot(x='index',y='Age_desc', data=cn, palette='Paired')

        plt.setp(ax_7.get_xticklabels(), horizontalalignment='right')
        plt.xlabel("Age Description - Based on Test Taker\'s Demographics Data", fontsize= 28, fontstyle='italic')
        plt.ylabel('Frequency(%count of total)', fontsize= 28, fontstyle='italic')
        plt.title("Test Taker\'s Demographics - Age Description Analysis", fontsize=30, fontweight='bold', y=1.02)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=28)
        plt.grid(True)
        for i in ax_7.patches:
            ax_7.annotate(format(i.get_height(), '.2f')+"%", (i.get_x() + i.get_width() / 2.\
                                                            , i.get_height()), ha = 'center'\
                        , va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=28)
        plt.tight_layout(pad=2.0)
        plt.show()

        
    @staticmethod
    def generate_random_samples(c=['red','green','blue','yellow','purple','orange','olive','brown'],
                                n=2):
        """
        Samples a number of distinct colors chosen from an input color list.

        Parameters:
        c (List[str], optional): List of colors, defaults to ['red','green','blue','yellow','purple','orange','olive','brown']
        n (int, optional): Number of distinct colors to return, defaults to 2

        Returns:
        List[str]: A list containing n randomly sampled distinct colors chosen from the input color list c
        """
        # Checking if number of samples are less than or equal to total number of elements in c
        if n <= len(c):
            return random.sample(c, n)
        else:
            return "Error: Number of samples requested is greater than the number of elements in the list."
        
    @classmethod
    def create_barplots_cat_binary(cls, df, categorical_binary_features):
        """
        Create barplots for categorical binary features in the ASD dataset.

        Parameters:
        df (pd.DataFrame): The dataset to create the barplots from
        categorical_binary_features (List[str]): List of categorical binary features to create barplots for

        Outputs:
        Barplots for each feature in the list of categorical binary features
        """
        subplot_num = (len(categorical_binary_features))*100+11
        plt.figure(figsize=(12,len(categorical_binary_features)*8))
        for i in categorical_binary_features:
            plt.subplot(subplot_num)
            # Color
            chosen_color_pair = cls.generate_random_samples()  # Calling static method within the current 'magic_helper' class 
            (df[i].value_counts(normalize=True)*100).round(2).plot(kind='bar', 
                                                                   rot=0, 
                                                                   color=chosen_color_pair)
            plt.title(f"'{i}' - Based on Test Taker\'s Demographics Data", fontsize=20, fontweight='bold', pad=10)
            plt.xlabel(i, fontsize=18, fontstyle='italic')
            plt.ylabel(f"Frequency (%)", fontsize=18, fontstyle='italic')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.ylim(0,100)
            plt.grid(True)
            subplot_num+=1
        plt.autoscale()
        plt.tight_layout(pad=2.0)
        plt.show()


    @staticmethod
    def plot_cat_vars_vs_target(df, cat_features, target_col="Class/ASD"):
        """
        Create a visualization of categorical variables against the target variable.

        Parameters:
        df (pandas.DataFrame): The dataset to create the visualization from
        cat_features (List[str]): List of categorical variables to visualize against the target variable
        target_col (str): The name of the target variable 

        Outputs:
        A visualization of categorical variables against the target variable
        """
        for cat_feature in cat_features:
            plt.figure(figsize=(16,6))
            my_df= df[[cat_feature,target_col]]
            sns.countplot(x = cat_feature, hue = target_col,data = my_df, palette=("plasma"))
            plt.title(f"'{cat_feature}' - Based on Test Taker\'s Demographics Data", 
                                                  fontweight='bold',fontsize=20, pad=10)
            plt.xlabel(cat_feature, fontsize=18, fontstyle='italic')
            plt.ylabel(f"Count", fontsize=18, fontstyle='italic')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tight_layout()
            plt.show()


    @staticmethod
    def plot_numeric_vars_vs_target(df, numeric_features):
        """
        Plots the numeric variables against the target variable 'Class/ASD'
        """
        for num_feature in numeric_features:
            plt.figure(figsize=(15,5),dpi=80)
            ax = sns.countplot(data=df, x=num_feature, hue="Class/ASD")
            ax.set_xlabel(ax.get_xlabel(), fontdict={'size': 18})
            ax.set_ylabel(ax.get_ylabel(), fontdict={'size': 18})
            plt.show()
            
            
    @staticmethod
    def plot_heatmap(df):
        """
        Plots a heatmap based on the correlation matrix representation of the input dataset (df), 
        indicating the correlation between numeric variables
        """
        plt.figure(figsize = (12,8), dpi=100)
        corr_matrix_new = df.corr()
        my_mask_1 = np.triu(np.ones_like(corr_matrix_new, dtype=np.bool))
        f, ax_corr1 = plt.subplots(figsize=(15, 15), dpi=100)
        ax_corr1 = sns.heatmap(corr_matrix_new, cmap= 'YlGnBu', cbar_kws={"shrink": .5}, vmin= -1, vmax=1, center=0,
                    square=True, mask=my_mask_1, annot=True)
        plt.xticks(fontsize=13, rotation=30)
        plt.yticks(fontsize=13, rotation=30)
        plt.title("Heatmap: Correlation Matrix", y=1.02, fontsize=25, fontweight='bold')
        plt.tight_layout()
        plt.autoscale()
        plt.show()

        

    @staticmethod
    def plot_confusion_matrix(cm):
        """
        Plots and prints the confusion matrix based on true and predicted labels for the target variable 'Class/ASD'
        """
        classes=['Non-ASD','ASD']
        cmap=plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix', fontsize=16)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes, rotation=90)
        thresh = cm.max() / 2.
        for i in range (cm.shape[0]):
            for j in range (cm.shape[1]):
                plt.text(j, i, cm[i, j],horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label', fontsize= 14)
        plt.xlabel('Predicted label', fontsize=14)

    
    @classmethod
    def print_model_metrics(cls, y_test,y_pred):
        """
        Evaluates a model's performance by taking true and predicted labels as inputs and prints the evaluation metrics
        """
        print(" Model Stats Scores Summary : ")
        cp = confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(5,5))
        cls.plot_confusion_matrix(cp)
        plt.show()    
        

    @staticmethod
    def plot_roc_auc_curve(fpr, tpr, roc_auc):
        """
        Plots the ROC-AUC curve given the inputs:
            fpr: False positive rate
            tpr: True positive rate
            roc_auc: ROC-AUC score
        """
        print(f"ROC for test dataset {round(roc_auc*100,3)}%")  
        plt.figure(figsize=(5,5))
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.plot(fpr,tpr,'b',label="Test ROC_AUC="+str(round(roc_auc,3)))
        plt.plot([0,1],[0,1],'r--')
        plt.ylabel('True Positive Rate (TPR)')
        plt.xlabel('False Positive Rate (FPR)')
        plt.legend(loc='lower right')
        plt.show()

        
    @staticmethod
    def best_cross_val_results(clf, model_name):
        """
        Takes the best trained model/estimator and model name as inputs, 
        returns the evaluation results from K-Fold cross-validation, 
        and the optimal hyperparameters.
        """
        print('\033[1m'+"*"*100+f"\nModel Name: {model_name}");print("*"*100)
        print('\033[1m'+"\nEvaluation results from cross-validation and optimal hyperparameters"+'\033[0m')
        # Best Model and optimal hyperparameters
        print('\033[1m'+"Best Estimator:\n"+'\033[0m', clf.best_estimator_._final_estimator)
        print('\033[1m'+f"Best Cross-Validation Accuracy: {100*clf.best_score_:.3f}%");print('\033[0m')
        print('\033[1m'+"Best (Optimal) Parameters:\n", clf.best_params_);print('\033[0m')


    @classmethod
    def prediction_evaluation_results(cls,clf,X_train,y_train,X_test,y_test,model_name,results_df):
        """
        Evaluates the performance of the best estimator/model obtained after 
        cross-validation and hyperparameter tuning using unseen test dataset.
        Parameters:
            clf: best estimator/model 
            X_train: input features from the train dataset
            y_train: true labels from the train dataset
            X_test: input features from the test dataset
            y_test: true labels from the test dataset (for evaluation)
            model_name: name of the model/estimator
            results_df: dataframe containing model metric scores
        Prints desired metrics and returns:
            results_store_df: pandas dataframe containing the evaluation metric scores 
        """
        print('\033[1m'+"*"*100+"\n\nPrediction and Evaluation results: On Actual TEST SET"+'\033[0m')
        y_pred= clf.predict(X_test)                                             # Find predicted values
        y_pred_probs = clf.predict_proba(X_test)[:,1]                           # Find predicted probabilities
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)           # Precision and Recall Scores
        recall= metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1_score= metrics.f1_score(y_pred=y_pred, y_true=y_test)                    # f1_score
        test_roc_auc = metrics.roc_auc_score(y_score= y_pred_probs, y_true=y_test)  # Test ROC_AUC
        print('\033[1m'+f"\nTest ROC_AUC: {test_roc_auc*100:.3f}%")
        test_accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)        # test accuracy
        print('\033[1m'+f"Test Accuracy: {test_accuracy*100:.3f}%");print()
        print('\033[1m'+"Confusion Matrix"+'\033[0m')                                                   # print confusion matrix
        
        cls.print_model_metrics(y_test, y_pred)
        print('\033[1m'+"Classification Report"+'\033[0m')                                              # Print classification report
        print(classification_report(y_test, y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs)              # fpr, tpr and threshold
        threshold= thresholds[np.argmax(tpr-fpr)]                                   # Find the optimal threshold value
        print('\033[1m'+f"Optimal Threshold: {threshold:.4f}");print('\033[0m')
        
        cls.plot_roc_auc_curve(fpr, tpr, test_roc_auc)      # Plots ROC_AUC curve for test dataset (using defined method)
        # Store values
        cross_val_acc = f"{clf.best_score_*100:.3f}%"
        test_accuracy = f"{test_accuracy*100:.3f}%"
        test_roc_auc = f"{test_roc_auc*100:.3f}%"
        threshold = f"{threshold:.4f}"
        precision = f"{precision*100:.3f}%"
        recall = f"{recall*100:.3f}%"
        f1_score = f"{f1_score*100:.3f}%"
        ## Store results 
        results_store_df= pd.DataFrame({'Model': [model_name],
                                        'Cross-Validation Accuracy': [cross_val_acc],
                                        'Test Accuracy': [test_accuracy], 
                                        'ROC_AUC_Test': [test_roc_auc],'Threshold': [threshold],
                                        'Precision': [precision],'Recall': [recall],
                                        'F1 Score':[f1_score]})
        return results_store_df