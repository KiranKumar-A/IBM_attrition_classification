#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhuoqinyu
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
import xgboost

## Data analysis
data=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
data["Attrition"][data['Attrition']=="No"].count()
#237 Yes and 1233 No imbalanced
#data.head()
#data.isnull().any()

### separate numerical and categorical variables
data_num=data.select_dtypes(include=['int64'])
data_cat=data.select_dtypes(include=['object'])
### heatmap of numerical variables:
plt.figure(figsize=(45,43))
foo=sns.heatmap(data_num.corr(),vmax=0.8,square=True)
plt.xticks(rotation=90)
plt.yticks(rotation=45)
plt.tight_layout()

##mapping to target label
# Define a dictionary for the target mapping
target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
data["target_numerical"] = data["Attrition"].apply(lambda x: target_map[x])

## set of scatterplots
sns.set()
cols=[u'Age', u'DailyRate',  u'JobSatisfaction',
       u'MonthlyIncome', u'PerformanceRating',
        u'WorkLifeBalance', u'YearsAtCompany', u'target_numerical']
sns.pairplot(data[cols],size=2.5,palette='seismic', diag_kind = 'kde',diag_kws=dict(shade=True))
plt.show()
## Feature engineering
#data=data.drop(["target_numerical"],axis=1)
categorical=[]
for col, value in data.iteritems():
    if value.dtype=='object':
        categorical.append(col)
numerical=data.columns.difference(categorical)
data_cat=data[categorical]
data_cat=data_cat.drop(["Attrition"],axis=1)
data_cat=pd.get_dummies(data_cat)
data_num=data[numerical]
target=data["target_numerical"]
data_num=data_num.drop(["target_numerical"],axis=1)
data_final=pd.concat([data_num,data_cat],axis=1)
#==============================================================================
# ##Machine learning 
#==============================================================================
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
train,test,target_train,target_val=train_test_split(data_final,target,
                                                    train_size=0.8,random_state=0)
#SMOTE oversample
oversampler=SMOTE(random_state=0)
smote_train,smote_target=oversampler.fit_sample(train,target_train)
#==============================================================================
#### Random forest classifier
#seed=0
#rf_params={
#    'n_jobs': -1,
#    'n_estimators': 800,
#    'warm_start': True, 
#    'max_features': 0.3,
#    'max_depth': 9,
#    'min_samples_leaf': 2,
#    'max_features' : 'sqrt',
#    'random_state' : seed,
#    'verbose': 0
#}
#rf = RandomForestClassifier(**rf_params)## unpacking rf_params
#rf.fit(smote_train,smote_target)
#print("Fitting of RF as finished")
#rf_predictions=rf.predict(test)
#print("Predictions finished")
#accuracy_score(target_val,rf_predictions)
#
## plot feature importance
#pldata=pd.DataFrame(data=rf.feature_importances_,columns=["importances"])
#pldata["features"]=data_final.columns
#pldata=pldata.sort_values("importances",ascending=False)
#sns.set(style="whitegrid")
#f, ax = plt.subplots(figsize=(6, 15))
#sns.set_color_codes("pastel")
#sns.barplot(x="importances", y="features", data=pldata,
#            label="importances", color="b")
#sns.despine(left=True, bottom=True)
#==============================================================================

#### Decision tree model and graph
#from sklearn import tree
#from IPython.display import Image as PImage
#from subprocess import check_call
#from PIL import Image, ImageDraw, ImageFont
#import re
#
#decision_tree = tree.DecisionTreeClassifier(max_depth = 4)
#decision_tree.fit(train, target_train)
#
## Predicting results for test dataset
#y_pred = decision_tree.predict(test)
#
## Export our trained model as a .dot file
#with open("tree1.dot", 'w') as f:
#     f = tree.export_graphviz(decision_tree,
#                              out_file=f,
#                              max_depth = 4,
#                              impurity = False,
#                              feature_names = data_final.columns.values,
#                              class_names = ['No', 'Yes'],
#                              rounded = True,
#                              filled= True )
#        
##Convert .dot to .png to allow display in web notebook
#check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
#
## Annotating chart with PIL
#img = Image.open("tree1.png")
#draw = ImageDraw.Draw(img)
#img.save('sample-out.png')
#PImage("sample-out.png")
#==============================================================================
#### Gradient boosted classifier
data_final=data_final.drop('EmployeeNumber',1)
train,test,target_train,target_val=train_test_split(data_final,target,
                                                    train_size=0.8,random_state=0)
#SMOTE oversample
oversampler=SMOTE(random_state=0)
smote_train,smote_target=oversampler.fit_sample(train,target_train)
### remove feature of EmplyeeNumber because GBC thinks it is important, which doesn't seems right
seed=0
gb_params ={
    'n_estimators': 500,
    'max_features': 0.9,
    'learning_rate' : 0.2,
    'max_depth': 11,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}
gb = GradientBoostingClassifier(**gb_params)
# Fit the model to our SMOTEd train and target
gb.fit(smote_train, smote_target)
# Get our predictions
gb_predictions = gb.predict(test)
print("Predictions have finished")
print(accuracy_score(target_val, gb_predictions))
# feature importance
pldata=pd.DataFrame(data=gb.feature_importances_,columns=["importances"])
pldata["features"]=data_final.columns
pldata=pldata.sort_values("importances",ascending=False)
sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(6, 15))
sns.set_color_codes("pastel")
sns.barplot(x="importances", y="features", data=pldata,
            label="importances", color="b")
sns.despine(left=True, bottom=True)