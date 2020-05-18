# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:03:08 2020

@author: JAE6KOR
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

upvotes_data_loc = r'D:\Seaborn tutorial\Predict number of upvotes'
train_data = pd.read_csv(os.path.join(upvotes_data_loc, 'train_data.csv'))

#df = train_data.iloc[:1000]
df = train_data.copy()
#%%

"data pre-processing"
summary = df.describe()

"missing data summary"
summary_na= [df[x].isna().sum() for x in df.columns] # we don't have any nan values in this data
#%%
"find correlation matrix"
df_corr = df.corr()

"generating heatmap using df_corr"
sns.heatmap(df_corr) # we don't see any var-pair haivng > 0.5 correlation, hence we'll be directly going for model building
#%%
"splitting train-test data"
from sklearn.model_selection import train_test_split

"Converting categorical str to continuous form usig dummy variables"
df1 = pd.concat([df.loc[:, df.columns!='Tag'], pd.get_dummies(df[['Tag']])], axis=1)

input_features = df1.drop(columns=['ID','Username', 'Upvotes']).values
target = df1['Upvotes'].values

X_train, X_val, y_train, y_val = train_test_split(input_features, target, random_state=10)


"ML models"
# we'll try 6 different models to get best model with best hyper-parameters
# will use gridsearchcv o randomizedsearchcv for parameter optimization

from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoLars
from xgboost import XGBRegressor
from sklearn.svm import SVR

poly5 = PolynomialFeatures(5)

llr = LassoLars()
knn = KNeighborsRegressor()
rfr = RandomForestRegressor()
xgbr = XGBRegressor()
svr = SVR()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as mse

models = [llr, knn, svr, rfr, xgbr]
grid_params = {llr: {'alpha':[0,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]},
                 knn:{'n_neighbors':[1,3,5,11,15,17,20,30,40,50,70],'weights':['uniform','distance'],'metric':['euclidean','manhattan']},
                 svr:{'C': [0.1, 1, 10, 100, 1000],'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                  rfr: {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)],# Number of trees in random forest
                                                 'max_features': ['auto', 'sqrt',None],# Number of features to consider at every split
                                                 'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],# Maximum number of levels in tree
                                                 'min_samples_split': [2, 5, 10],# Minimum number of samples required to split a node
                                                 'min_samples_leaf': [1, 2, 4],# Minimum number of samples required at each leaf node
                                                 'bootstrap': [True, False]},# Method of selecting samples for training each tree
                 xgbr:{'eta':[0.1,0.3,0.5,0.7],'gamma':[0,10,100],'max_depth':[1,3,5,9,11]}}                                     

#%%
import time
scoring='neg_mean_squared_error'
cv_results_list = []
best_params_list = []
best_cv_rmse_list = []
val_rmse_list = []
train_time_list = []
for model in models[-1:]:
    num_iter = 1
    st = time.time()
#    model = GridSearchCV(knn, grid_params[knn], n_jobs=-1, scoring=scoring) # not performing gridsearchcv as it takes lot time
    total_param_combinations = np.prod([len(grid_params[model][par]) for par in list(grid_params[model].keys())])
    
    model = RandomizedSearchCV(estimator=model, n_iter=num_iter, cv=3, scoring=scoring,n_jobs=-1, param_distributions=grid_params[model], refit=True)
    
    if model==llr:
        X_train1 = poly5.fit_transform(X_train)
        model.fit(X_train1, y_train)
    else:
        model.fit(X_train, y_train)
    
    
    cv_results_list+=[model.cv_results_]
    
    best_cv_rmse1 = np.sqrt(-max(model.cv_results_['mean_test_score']))
    best_cv_rmse_list+=[best_cv_rmse1]
    
    best_params_list+=[model.best_params_]
    
    train_time1 = str(round(time.time()-st,2))+' seconds'
    train_time_list+=[train_time1]
    if model==llr:
        X_val1 = poly5.transform(X_val)
        y_pred = model.predict(X_val1)
    else:
        y_pred = model.predict(X_val)
    
    rmse1 = np.sqrt(mse(y_val,y_pred))
        
    val_rmse_list+=[rmse1]
    
    print('total_cv_time:',train_time1)
    print('avg_cv_rmse:',best_cv_rmse1)
    
    print('val_rmse:', rmse1)


#%%