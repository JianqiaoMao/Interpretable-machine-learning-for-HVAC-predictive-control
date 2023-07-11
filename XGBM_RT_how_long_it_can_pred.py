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
    
def print_eq(coef, intercept):
    eq="y="
    for i in range(len(coef)):
        eq += "{:.3f}*x{}".format(coef[i], i+1) 
    eq += "{:.3f}".format(intercept)   
    print(eq)
        

# R_res, R_unres, df    
def chow_test(X, y, break_point, alpha = 0.05):
    sub_lr_models = []
    for i in range(len(break_point)+1):
        tempData = LinearRegression()
        exec("sub_lr_models.append(tempData)")
    
    R_unres = 0
    k = X.shape[1]+1
    df = -(len(break_point)+1)*k + X.shape[0]
    
    for i in range(len(break_point)+1):
        if i==0:
            X_train, y_train = X.loc[:break_point[i]], y.loc[:break_point[i]]
        elif i<len(break_point):
            X_train, y_train = X.loc[break_point[i-1]:break_point[i]], y.loc[break_point[i-1]:break_point[i]]
            X_train, y_train = X_train.iloc[1:], y_train.iloc[1:]
        else:
            X_train, y_train = X.loc[break_point[i-1]:], y.loc[break_point[i-1]:]
            X_train, y_train = X_train.iloc[1:], y_train.iloc[1:]        
        sub_lr_models[i].fit(X_train, y_train)
        preds_sub = sub_lr_models[i].predict(X_train)
        R_unres += sum((preds_sub-y_train)**2)
        sub_coef = sub_lr_models[i].coef_
        sub_intercept = sub_lr_models[i].intercept_
        print("*"*50)
        print("Period: {}-{}".format(X_train.index[0], X_train.index[1]))
        print_eq(sub_coef, sub_intercept)
        print("Sum of Square Residual (SSR): {:.2f}".format(R_unres))
        
    
    global_lr_model = LinearRegression()
    global_lr_model.fit(X,y)
    preds = global_lr_model.predict(X)
    
    global_coef = global_lr_model.coef_
    global_intercept = global_lr_model.intercept_
    R_res = sum((preds-y)**2)
    
    print("*"*50)
    print("Period: {}-{}".format(X.index[0], X.index[1]))
    print_eq(global_coef, global_intercept)
    print("Sum of Square Residual (SSR): {:.2f}".format(R_res))
    print("*"*50)
    
    F = ((R_res-R_unres)/k)/(R_unres/df)
    p_value = stats.f.cdf(F, k, df)
    if p_value < alpha:
        print("There are structual changes at the selected time points with confidence_interval={}".format(1-alpha))
    else:
        print("There are not structual changes at the selected time points with confidence_interval={}".format(1-alpha))
    
    return F, [k,df], p_value, sub_lr_models, global_lr_model

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
    if x.shape[0]%pred_length!=0:
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
    else:
        for i in range(int(x.shape[0]/pred_length)):
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
    return y_pred, x

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

def FFT_spectrum(series):
    sampling_t = 300
    F_samp = 1/sampling_t
    n = series.shape[0]
    k = np.arange(n)
    T = n/F_samp
    twoside_frq_range = k/T
    oneside_frq_range  = twoside_frq_range[range(int(n/2))]     
    
    spectrum = np.fft.fft(series)/n
    twoSide_amp = np.abs(spectrum)
    oneSide_amp = twoSide_amp[range(int(n/2))]
    twoSide_angle = np.angle(spectrum)
    oneSide_angle = twoSide_angle[range(int(n/2))]
    
    return oneSide_amp, oneSide_angle, oneside_frq_range

def mbe_cal(y_true, y_pred):
    return 100/(len(y_true)-1)*sum((y_true -  y_pred))/np.mean(y_true)
#%% Load Dataset    

dataset = pd.read_csv("./Complete_Dataset.csv", usecols=[i+1 for i in range(7)])
dataset.loc[:,'Time'] = pd.to_datetime(dataset['Time'])
dataset = dataset.set_index('Time', drop=True)

roomTemp = dataset.loc[:,"RoomTemp_0103"]
features = dataset.drop(["SetTemp_0103", "RoomTemp_0103"], axis=1)

split_point = {"Train_begin": datetime.strptime("2017-12-08 15:39:36", '%Y-%m-%d %H:%M:%S'),
               "Train_end": datetime.strptime("2019-06-30 23:57:55", '%Y-%m-%d %H:%M:%S'),
               "Val_begin": datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'),
               "Val_end":  datetime.strptime("2019-10-10 23:56:41", '%Y-%m-%d %H:%M:%S'),
               "Test_begin": datetime.strptime("2019-10-11 00:06:41", '%Y-%m-%d %H:%M:%S'),
               "Test_end": datetime.strptime("2020-02-29 23:58:59", '%Y-%m-%d %H:%M:%S')}

#%% Feature Engineering

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

W_size = 6
features.loc[:,"OUTAIRHUMD_1h_MVA"] = move_avg(features.loc[:,"OUTAIRHUMD"], W_size)
features.loc[:,"OUTAIRTEMP_1h_MVA"] = move_avg(features.loc[:,"OUTAIRTEMP"], W_size)
features = features.drop(["OUTAIRHUMD", "OUTAIRTEMP"], axis=1)
MVA_RT_2h = move_avg(roomTemp, W_size)
features.loc[:,"RT_1h_HistoricalMVA"] = np.array([np.nan]+MVA_RT_2h)[:-1]

features=features.dropna()
roomTemp = roomTemp.iloc[1:]

#%% Dataset Spliting

X_train = features.loc[:split_point["Train_end"]]
y_train = roomTemp.loc[:split_point["Train_end"]]

X_val = features.loc[split_point["Val_begin"]:split_point["Val_end"]]
y_val = roomTemp.loc[split_point["Val_begin"]:split_point["Val_end"]]

X_test = features.loc[split_point["Test_begin"]:split_point["Test_end"]]
y_test = roomTemp.loc[split_point["Test_begin"]:split_point["Test_end"]]

#%% Regression XGBM
gbm_reg = XGBRFRegressor(max_depth=11, gamma=0.3,n_estimators=140)

gbm_reg.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
#joblib.dump(gbm_reg,"xgbm_reg.model")

#%% Metrics Over Predicting Time Length On Validation Set
mae_test_ot = []
mse_test_ot = []
mape_test_ot = []
r2_test_ot = []
mbe_test_ot = []
test_time_length = [1, 6, 12, 24, 48, 60, 72, 90, 120, 144, 216, 288, 360, 432, 504, 600]
for i in test_time_length:
    y_pred_test, _ =  queue_pred(X_test, gbm_reg, pred_length=i, window_size=W_size)
    y_pred_test = np.round(y_pred_test*2)/2
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    mbe = mbe_cal(y_test, y_pred_test)
    mae_test_ot.append(mae)
    mse_test_ot.append(mse)
    mape_test_ot.append(mape)
    r2_test_ot.append(r2)
    mbe_test_ot.append(mbe)
    print("Predicting Series Length(in Hour): {}, MAE: {:.4f}, MSE: {:.4f}, MAPE:{:.4f}, R2:{:.4f}, MBE:{:.4f}".format(i/6, mae, mse, mape, r2, mbe))
    
fig=plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16 
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.plot([i/6 for i in test_time_length], mae_test_ot, label="MAE", marker="o", c="royalblue", lw=3)
plt.plot([i/6 for i in test_time_length], mse_test_ot, label="MSE", marker="o", c="orange", lw=3)
for a,b,c in zip([int(10*i/60) for i in test_time_length],mae_test_ot, mse_test_ot):
    if a != 1:
        if a <=3:
            plt.text(a, b+0.03, '%.2f' % b, ha='center', va= 'bottom',fontsize=12)
            plt.text(a, c-0.08, '%.2f' % c, ha='center', va= 'bottom',fontsize=12)
        else:
            plt.text(a, b-0.08, '%.2f' % b, ha='center', va= 'bottom',fontsize=12)
            plt.text(a, c+0.03, '%.2f' % c, ha='center', va= 'bottom',fontsize=12)        
plt.xlabel("Time Length(Hour)")
plt.ylabel("MAE/MSE")
plt.grid(axis='x')
plt.xlim(-1,101)
plt.xticks([5*i for i in range(21)])
plt.legend(loc="lower right")
plt.title("MAE & MSE of XGBM Regressor's Prediction With Different Predicting Time Intervals")


#%% Result plotting
fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(111)
ax1.plot([i/6 for i in test_time_length], mae_test_ot, label="MAE", marker="o", c="royalblue", lw=3)
ax1.set_ylabel("MAE")
ax1.set_xlabel("Time Length(Hour)")
ax1.set_xticks([int(i/6) for i in test_time_length])
handles_1, labels_1 = ax1.get_legend_handles_labels()
for a,b in zip([int(10*i/60) for i in test_time_length],mae_test_ot):
    if a!= 0:
        plt.text(a, b+0.025, '%.2f' % b, ha='center', va= 'bottom',fontsize=12)

ax2 = ax1.twinx()
ax2.plot([i/6 for i in test_time_length], mse_test_ot, label="MSE", marker="o", c="orange", lw=3)
ax2.set_ylabel("MSE")
ax2.set_ylim(0, 6.5)
ax2.set_xlim(-1,101)
for i in range(len(test_time_length)):
    ax2.axvline(x = test_time_length[i]/6, ls="--", lw=0.5, color="grey")
handles_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(handles = handles_1+handles_2, loc = 'lower right')
ax2.set_title("MAE & MSE of XGBM Regressor's Prediction With Different Predicting Time Intervals")
for a,c in zip([int(10*i/60) for i in test_time_length], mse_test_ot):
    if a != 0:
        plt.text(a, c-0.25, '%.2f' % c, ha='center', va= 'bottom',fontsize=12)

