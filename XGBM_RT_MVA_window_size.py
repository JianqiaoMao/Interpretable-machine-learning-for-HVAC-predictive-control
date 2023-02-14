# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 01:28:45 2021

@author: Mao Jianqiao
"""
#%% Configuration
import numpy as np
import pandas as pd
from xgboost import XGBRFRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import holidays
from scipy import stats
import joblib
import lime
import lime.lime_tabular
import random
from statsmodels.tsa.stattools import ccf

def form_holiday_calendar():
    holidays_df = pd.DataFrame([], columns = ['ds','holiday'])
    ldates = []
    lnames = []
    for date, name in sorted(holidays.GR(years=np.arange(2017, 2020 + 1)).items()):
        ldates.append(str(date))
        lnames.append(name)
        
    holidays_df.loc[:,'ds'] = ldates
    holidays_df.loc[:,'holiday'] = lnames
    holidays_df.holiday.unique()
    
    return holidays_df.iloc[9:35]

def evaluate(y_pred, y_true, mode):
    if mode == "cla":
        acc = accuracy_score(y_true, y_pred)
        
        prec_ma = precision_score(y_true, y_pred,average='macro')
        recall_ma = recall_score(y_true, y_pred,average='macro')
        f1score_ma = f1_score(y_true, y_pred,average='macro')
        
        prec_w = precision_score(y_true, y_pred,average='weighted')
        recall_w = recall_score(y_true, y_pred,average='weighted')
        f1score_w = f1_score(y_true, y_pred,average='weighted')
        
        print("-"*50)
        print("Acc: %.4f" %(acc))
        print("*"*50)
        print("Weighted Precision: %.4f" %(prec_w))
        print("Weighted recall: %.4f" %(recall_w))
        print("Weighted f1-score: %.4f" %(f1score_w))
        print("*"*50)
        print("Macro-averaged Precision: %.4f" %(prec_ma))
        print("Macro-averaged recall: %.4f" %(recall_ma))
        print("Macro-averaged f1-score: %.4f" %(f1score_ma))
        print("-"*50)    
        
        conf_matrix(y_true, y_pred)
        
        return acc, prec_ma, recall_ma, f1score_ma, prec_w, recall_w, f1score_w

    elif mode== "reg":
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print("-"*50)
        print("MSE: %.6f" %(mse))
        print("MAE: %.6f" %(mae))
        print("MAPE: %.6f" %(mape))
        print("R-square: %.6f" %(r2))
        print("-"*50) 
        return mse, mae, mape, r2

def conf_matrix(y_true, y_pred, fig_name="Confision Matrix", axis_rename=None):
    confMat=confusion_matrix(y_true, y_pred)
    confMat=pd.DataFrame(confMat)
    if axis_rename==None:
        None
    else:
        confMat=confMat.rename(index=axis_rename,columns=axis_rename)
    plt.figure(facecolor='lightgray')
    plt.title(fig_name, fontsize=20)
    ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
    ax.set_xlabel('Predicted Class', fontsize=14)
    ax.set_ylabel('True Class', fontsize=14)
    plt.show()     
 

def queue_pred(x_input, model, pred_length, window_size):
    x = x_input.copy()
    y_pred = []
    for i in range(int(x.shape[0]/pred_length)+1):
        if i!= int(x.shape[0]/pred_length):
            MVA_record = [x.iloc[i*pred_length,-1]]
            for j in range(pred_length-1):
                pred_j = model.predict(np.array(x.iloc[i*pred_length+j,:]).reshape(1,-1))
                y_pred.append(pred_j[0])
                MVA_record.append(pred_j[0])
                MVA = np.mean(MVA_record[-window_size:])
                x.iloc[i*pred_length+j+1,-1] = MVA
            pred_j = model.predict(np.array(x.iloc[(i+1)*pred_length-1,:]).reshape(1,-1))
            y_pred.append(pred_j[0])  
        else:
            MVA_record = [x.iloc[i*pred_length,-1]]
            for j in range(x.shape[0]-pred_length*int(x.shape[0]/pred_length)-1):
                pred_j = model.predict(np.array(x.iloc[i*pred_length+j,:]).reshape(1,-1))
                y_pred.append(pred_j[0])
                MVA_record.append(pred_j[0])
                MVA = np.mean(MVA_record[-window_size:])               
                x.iloc[i*pred_length+j+1,-1] = MVA
            if x.shape[0]%pred_length != 0:
                pred_j = model.predict(np.array(x.iloc[i*pred_length+x.shape[0]-pred_length*int(x.shape[0]/pred_length)-1,:]).reshape(1,-1))
                y_pred.append(pred_j[0])
    y_pred = np.array(y_pred).reshape(-1)     
    x = pd.DataFrame(x, index=x_input.index, columns=x_input.columns)
    return y_pred

def move_avg(series,window_size):
    MVA = []
    if series.shape[0]<=window_size:
        for i in range(series.shape[0]-1):
                MVA.append(np.mean(series[:i+1]))      
    else:
        for i in range(window_size-1):
                MVA.append(np.mean(series[:i+1]))
    for i in range(window_size, series.shape[0]+1):
        MVA.append(np.mean(series[i-window_size:i]))
    return MVA

#%% Load Dataset    

dataset = pd.read_csv("./Complete_Dataset.csv", usecols=[i+1 for i in range(7)])
dataset.loc[:,'Time'] = pd.to_datetime(dataset['Time'])
dataset = dataset.set_index('Time', drop=True)

split_point = {"Train_begin": datetime.strptime("2017-12-08 15:39:36", '%Y-%m-%d %H:%M:%S'),
               "Train_end": datetime.strptime("2019-06-30 23:57:55", '%Y-%m-%d %H:%M:%S'),
               "Val_begin": datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'),
               "Val_end":  datetime.strptime("2019-10-10 23:56:41", '%Y-%m-%d %H:%M:%S'),
               "Test_begin": datetime.strptime("2019-10-11 00:06:41", '%Y-%m-%d %H:%M:%S'),
               "Test_end": datetime.strptime("2020-02-29 23:58:59", '%Y-%m-%d %H:%M:%S')}

#%% Test Different MVA window size

window_sizes = [0, 1, 3, 6, 9, 12, 18, 24, 30, 36, 48, 60, 72, 90, 108, 126, 144]

MSE_train = []
MAE_train = []
R2_train = []
MAPE_train = []

MSE_val = []
MAE_val = []
R2_val = []
MAPE_val = []

XGBMs=[]

roomTemp = dataset.loc[:,"RoomTemp_0103"]
features = dataset.drop(["SetTemp_0103", "RoomTemp_0103"], axis=1)
features.loc[:,"Quarter"] = features.index.quarter
features.loc[:,"Month"] = features.index.month
features.loc[:,"WeekDay"] = features.index.weekday
features.loc[:,"Hour"] = features.index.hour
features.loc[:,"OOS_OC"] = (features.loc[:,"Hour"]>=9) & (features.loc[:,"Hour"]<=18) & (features.loc[:,"OnOffState_0103"]==1)
features.loc[:,"OOS_OC"] = features.loc[:,"OOS_OC"].map(lambda x:int(x))

features.loc[:,"Weekend?"] = features.loc[:,"WeekDay"]>=5
Greece_holidays = form_holiday_calendar()
features.loc[:,"Holiday?"] = False
holiday_index = []
features_dates = features.index.date
for i in range(len(features.index)):
    if str(features_dates[i]) in list(Greece_holidays.loc[:, "ds"]):
        holiday_index.append(features.index[i])
features.loc[holiday_index ,"Holiday?"] = True
features.loc[:,"Holiday?"] = features.loc[:,"Holiday?"] | features.loc[:,"Weekend?"]
features.loc[:,"Holiday?"] = features.loc[:,"Holiday?"].map(lambda x:int(x))
features = features.drop("Weekend?", axis=1)

for W_size in window_sizes:

    FEATURES = features.copy()
    RT = roomTemp.copy()
    
    if W_size != 0:
    
        FEATURES.loc[:,"OUTAIRHUMD_MVA"] = move_avg(FEATURES.loc[:,"OUTAIRHUMD"], W_size)
        FEATURES.loc[:,"OUTAIRTEMP_MVA"] = move_avg(FEATURES.loc[:,"OUTAIRTEMP"], W_size)
        FEATURES = FEATURES.drop(["OUTAIRHUMD", "OUTAIRTEMP"], axis=1)
        MVA_RT = move_avg(RT, W_size)
        FEATURES.loc[:,"RT_HistoricalMVA"] = np.array([np.nan]+MVA_RT)[:-1]
        
        FEATURES=FEATURES.dropna()
        RT = RT.iloc[1:]

    X_train = FEATURES.loc[:split_point["Train_end"]]
    y_train = RT.loc[:split_point["Train_end"]]
    
    X_val = FEATURES.loc[split_point["Val_begin"]:split_point["Val_end"]]
    y_val = RT.loc[split_point["Val_begin"]:split_point["Val_end"]]
    
    X_test = FEATURES.loc[split_point["Test_begin"]:split_point["Test_end"]]
    y_test = RT.loc[split_point["Test_begin"]:split_point["Test_end"]]
    
    gbm_reg = XGBRFRegressor(max_depth=11, gamma=0.3,n_estimators=140)
    gbm_reg = gbm_reg.fit(X_train, y_train)
    XGBMs.append(gbm_reg)
    
    print("_"*75)
    print("XGBM with MVA window size: {}".format(W_size))

    print("*"*50)
    print("Valdation Phase")
    if W_size !=0:
        y_pred_val = queue_pred(X_val, gbm_reg, pred_length=48, window_size = W_size)
    else:
        y_pred_val = gbm_reg.predict(X_val)
    mse_val, mae_val, mape_val, r2_val = evaluate(y_pred_val, y_val, mode="reg")
    MSE_val.append(mse_val)
    MAE_val.append(mae_val)
    R2_val.append(r2_val)
    MAPE_val.append(mape_val)

#%% Performance Conclusion Ploting

fig = plt.figure()
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18 
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
ax1 = fig.add_subplot(111)
lns1 = ax1.plot(window_sizes, MAE_val,  "o-", c="royalblue", ms=6, label="MAE", lw=3)
ax1.set_ylabel("MAE")
ax1.set_xlabel("MVA Window Width (Hour)")
for a,b in zip(window_sizes,MAE_val):
    if a != 12 and a!= 6:
        ax1.text(a, b+0.01, '%.4f' % b, ha='center', va= 'bottom',fontsize=13)
    else:
        ax1.text(a, b-0.02, '%.4f' % b, ha='center', va= 'bottom',fontsize=13)
ax2 = ax1.twinx()
lns2 = ax2.plot(window_sizes, R2_val, "o-", c="darkorange", ms=6, label="R-square", lw=3)
ax2.set_ylabel("R-square")
lns_mul = lns1+lns2
labels = [l.get_label() for l in lns_mul ]
for a,b in zip(window_sizes,R2_val):
    if a!=1:
        if a!=12 and a!=6:
            ax2.text(a, b-0.01, '%.4f' % b, ha='center', va= 'bottom',fontsize=13)
        else:
            ax2.text(a, b+0.004, '%.4f' % b, ha='center', va= 'bottom',fontsize=13)
ax2.legend(lns_mul, labels, loc="center right")
ax2.set_xticks([0, 3, 6, 9, 12, 18, 24, 30, 36, 48, 60, 72, 90, 108, 126, 144])
ax2.set_xticklabels( ["0", "0.5", "1", "1.5"]+[str(int(i/6)) for i in window_sizes[5:]])
ax2.set_xlim(-5,149)
ax2.set_title("XGBM Regressor Performance Comparison for Different MVA Window Widths")
