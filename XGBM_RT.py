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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
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

def print_eq(coef, intercept, feature_names):
    eq="y="
    for i in range(len(coef)):
        if coef[i]>=0:            
            eq += "{:.3f}*".format(coef[i], i+1)+feature_names[i]
        else:
            eq += "({:.3f})*".format(coef[i], i+1) +feature_names[i]
        if i!=len(coef-1):
            eq += "+"
    eq += "{:.3f}".format(intercept)   
    print(eq)

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

def local_explianer(model, X_train, X_test, y_test, random=True, sample_indicies=None):
    if random:
        sample_indicies = [np.random.randint(0, X_test.shape[0])]
    else:
        if sample_indicies is None:
            raise ValueError("You have to assign a list or a int for 'sample_indicies' or set 'random' to True to randomly observe a sample.")
        else:
            sample_indicies = list(sample_indicies)
            
    feature_name_brief = []
    for i in range(len(X_test.columns)):
        if "_0103" in X_test.columns[i]:
            feature_name_brief.append(X_test.columns[i].split("_0103")[0])
        else:
            feature_name_brief.append(X_test.columns[i])    
    feature_name_brief[10] = "MVART"
    feature_name_brief[8] = "Outdoor Humid. (1h MVA)"
    feature_name_brief[9] = "Outdoor Temp. (1h MVA)"
            
    for sample in sample_indicies:
        print("*"*50)
        print("Generating local explanation for sample:{} ...".format(sample))
        pred_RT = y_pred_test[sample]
        true_RT = y_test[sample]
        print("True Labels: {:.2f}".format(true_RT))
        print("Predicted Labels: {:.2f}".format(pred_RT))

        plt.figure()
        plt.title("Local Explanation of Sample {} (True: {:.2f}, Pred.: {:.2f})".format(sample, true_RT, pred_RT),fontsize=16)

        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train), mode="regression", feature_names=feature_name_brief, discretize_continuous=True)
    
        local_exp = explainer.explain_instance(np.array(X_test)[sample], model.predict, num_features=12)
        w_list = local_exp.as_list()
        name = [w_list[i][0].replace(".00", "") for i in range(len(w_list))]
        weight = [w_list[i][1] for i in range(len(w_list))]
        name, weight, abs_w = zip(*sorted(zip(name, weight, np.abs(weight)),key=lambda x: x[2], reverse=False))
        color = []
        for w in weight:
            if w<0:
                color.append("royalblue")
            else:
                color.append("orange")
                    
            plt.barh(name,weight,color=color)
            plt.grid(True, axis="x")
            plt.xlabel("Feature Weight")
            plt.ylabel("Feature Name")

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
MVA_RT_1h = move_avg(roomTemp, W_size)
features.loc[:,"RT_1h_HistoricalMVA"] = np.array([np.nan]+MVA_RT_1h)[:-1]

features=features.dropna()
roomTemp = roomTemp.iloc[1:]

#%% Dataset Spliting

X_train = features.loc[:split_point["Train_end"]]
y_train = roomTemp.loc[:split_point["Train_end"]]

X_val = features.loc[split_point["Val_begin"]:split_point["Val_end"]]
y_val = roomTemp.loc[split_point["Val_begin"]:split_point["Val_end"]]

X_test = features.loc[split_point["Test_begin"]:split_point["Test_end"]]
y_test = roomTemp.loc[split_point["Test_begin"]:split_point["Test_end"]]



#%% GridSearch
gbm_reg = XGBRFRegressor()

param = {"max_depth":[5,6,7,8,9,10,11,12,13,14,15],
          "gamma" : [0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2],
          "n_estimators": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 500]}

X_trainVal = pd.concat([X_train, X_val])
y_trainVal = pd.concat([y_train, y_val])

val_fold = np.zeros(X_trainVal.shape[0])
val_fold[:X_train.shape[0]] = -1

ps = PredefinedSplit(test_fold=val_fold)
reg_candidates = GridSearchCV(gbm_reg, param, scoring = 'neg_mean_squared_error',cv = ps)
searched_regs = reg_candidates.fit(X_trainVal, y_trainVal)
bestReg=reg_candidates.best_estimator_

print("The best param. is {}".format(reg_candidates.best_params_))
print("Best validation MSE: {:.4f}" .format(-searched_regs.best_score_))

#%% Regression XGBM
gbm_reg = XGBRFRegressor(max_depth=11, gamma=0.3,n_estimators=140)

gbm_reg.fit(X_train, y_train)
joblib.dump(gbm_reg,"./models/xgbm_reg_withHist.model")

#%% Evaluate XGBM_regression
#gbm_reg=joblib.load("xgbm_reg.model")

print("*"*75)
print("Training Phase")
y_pred_train, _ = queue_pred(X_train, gbm_reg, pred_length=48, window_size = W_size)
y_pred_train = np.round(y_pred_train*2)/2
train_result = evaluate(y_pred_train, y_train, mode="reg")

print("*"*75)
print("Valdation Phase")
y_pred_val, _=  queue_pred(X_val, gbm_reg, pred_length=48, window_size = W_size)
y_pred_val = np.round(y_pred_val*2)/2
val_result = evaluate(y_pred_val, y_val, mode="reg")

pred_val_date = pd.Series((X_val.index))
plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.plot(pred_val_date, y_pred_val,  label='Prediction')
plt.plot(pred_val_date, y_val,  label='Ground True')
plt.ylim(21,40)
plt.xlim(pred_val_date.iloc[0],pred_val_date.iloc[-1])
plt.legend()
plt.title("XGBM Regressor Predictions v.s Ground True")
plt.show()

#%% Metrics Over Predicting Time Length On Validation Set
mae_val_ot = []
mse_val_ot = []
test_time_length = [6, 12, 30, 48, 60, 72, 90, 120, 144, 216, 288, 360, 432, 504, 600]
for i in test_time_length:
    y_pred_val =  queue_pred(X_val, gbm_reg, pred_length=i, window_size=W_size)
    mae = mean_absolute_error(y_val, y_pred_val)
    mse = mean_squared_error(y_val, y_pred_val)

    mae_val_ot.append(mae)
    mse_val_ot.append(mse)
    print("Predicting Series Length(in Hour): {}, MAE: {:.4f}, MSE: {:.4f}".format(i/6, mae, mse))
    
fig=plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16 
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.plot([i/6 for i in test_time_length], mae_val_ot, label="MAE", marker="o", c="royalblue", lw=3)
plt.plot([i/6 for i in test_time_length], mse_val_ot, label="MSE", marker="o", c="orange", lw=3)
for a,b,c in zip([int(10*i/60) for i in test_time_length],mae_val_ot, mse_val_ot):
    plt.text(a, b-0.1, '%.2f' % b, ha='center', va= 'bottom',fontsize=12)
    plt.text(a, c+0.015, '%.2f' % c, ha='center', va= 'bottom',fontsize=12)
plt.xlabel("Time Length(Hour)")
plt.ylabel("MAE/MSE")
plt.grid(axis='x')
plt.xlim(-1,101)
plt.xticks([5*i for i in range(21)])
plt.legend(loc="lower right")
plt.title("MAE & MSE of XGBM Regressor With Different Predicting Length")

#%% Test Phase

print("*"*75)
print("Test Phase")
print("Re-training the model...")

X_trainVal = pd.concat([X_train, X_val])
y_trainVal = pd.concat([y_train, y_val])

gbm_reg = XGBRFRegressor(max_depth=11, gamma=0.3, n_estimators=140)
gbm_reg.fit(X_trainVal, y_trainVal)

print("*"*50)
print("Evaluating...")
y_pred_test, X_test_ = queue_pred(X_test, gbm_reg, pred_length=48, window_size = W_size)
y_pred_test_round = np.round(y_pred_test*2)/2
test_result = evaluate(y_pred_test, y_test, mode="reg")

pred_test_date = pd.Series((X_test.index))
plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.plot(pred_test_date, y_pred_test_round,  label='Prediction')
plt.plot(pred_test_date, y_test,  label='Ground True')
plt.ylim(13,33)
plt.xlim(pred_test_date.iloc[0],pred_test_date.iloc[-1])
plt.legend()
plt.title("XGBM Regressor Predictions v.s Ground True")
plt.show()

pred_all_date = pd.Series((features.loc[:split_point["Test_end"]].index))
preds_all = np.hstack((np.hstack((y_pred_train,y_pred_val)),y_pred_test_round))
y_all = np.hstack((np.hstack((y_train,y_val)),y_test))

plt.figure(figsize=(15,3.27))
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.plot(pred_all_date,y_all,  label='Ground True',linewidth=0.8)
plt.plot(pred_all_date, preds_all,  label='Prediction',linewidth=0.8)
plt.ylim(13,40)
plt.xlim(pred_all_date.iloc[0],pred_all_date.iloc[-1])
plt.vlines(pred_all_date[y_train.shape[0]-1],13,40,color="black",linestyle='--')
plt.fill_between(pred_all_date[y_train.shape[0]-1:y_train.shape[0]+y_val.shape[0]-2],13,40,facecolor='blue', alpha=0.3)
plt.vlines(pred_all_date[y_train.shape[0]+y_val.shape[0]-1],13,40,color="black",linestyle='--')
plt.fill_between(pred_all_date[y_train.shape[0]+y_val.shape[0]-1:],13,40,facecolor='red', alpha=0.3)
plt.legend(loc="upper left")
plt.title("XGBM Regressor Predictions v.s Ground True (With the Historical RT Feature)")
plt.show()


#%% Residual Analysis

residual = y_test - y_pred_test

# results = pd.concat([y_test.rename("y_true"), pd.Series(y_pred_test, index = y_test.index, name = 'y_pred_withHist')], axis=1)
# results.to_csv("./withHist_pred.csv")

results = pd.DataFrame(y_test.rename("y_true"))
results["y_pred_withoutHist_withoutOther"] = y_trainVal.mean()
results.to_csv("./withoutHist_withoutOther_pred.csv")

plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.title("Residual Distribution")
plt.hist(residual, bins=50)
plt.xlabel("Residual (Degree)")
plt.ylabel("Count")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
QQ_plot = stats.probplot(residual, plot=ax)

varr, mean, skew = np.var(residual), np.mean(residual), stats.skew(residual)
print("The prediction residual has variance:{:.2f}, mean:{:.2f}, skewness:{:.2f}".format(varr, mean, skew))

# Prediction-True Lag Analysis
order = 144*1 # 1 day
ccf_coef_forward = ccf(x=y_pred_test,y=y_test)[:order+1]
ccf_coef_backward = ccf(x=y_pred_test[::-1], y=y_test[::-1])[::-1][-order:]
ccf_coef = np.r_[ccf_coef_backward, ccf_coef_forward]
plt.figure()
order_range = [i for i in range(-order,order+1)]
plt.scatter(order_range, ccf_coef, marker="o")
plt.vlines(order_range, 0, ccf_coef)
plot_time_interval = 2 # in hour
plot_time_start = -int(order/(6*plot_time_interval))
plot_time_end = int(order/(6*plot_time_interval))+1
plt.xticks([6*plot_time_interval*i for i in range(plot_time_start,plot_time_end)],[i*plot_time_interval for i in range(plot_time_start,plot_time_end)])
plt.ylim(0,1)
plt.xlim(-order-1,order+1)

#%% Global Interpretation
# Feature Importance (Criterion: MSE)
gbm_fi = gbm_reg.feature_importances_
print("*"*50)
print("XGBM Regressor Feature Importance:")
for i in range(gbm_fi.shape[0]):
    print("{}: {:.6f}".format(X_train.columns[i], gbm_fi[i]))

w_list = gbm_fi
feature_names = [X_train.columns[i].replace("_0103", "") for i in range(len(w_list))]
feature_names[10] = "MVART"
feature_names[8] = "Outdoor Humid. (1h MVA)"
feature_names[9] = "Outdoor Temp. (1h MVA)"
weight = [gbm_fi[i] for i in range(len(w_list))]
name, weight = zip(*sorted(zip(feature_names, weight),key=lambda x: x[1], reverse=False))
plt.figure()
plt.title("Feature Importance for XGBM Regressor",fontsize=16)
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.barh(name,weight,color="royalblue")
plt.grid(True, axis="x")
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
for i in range(len(weight)):
    plt.text(weight[i]+0.03, name[i], "%.4f" % weight[i], ha='center', fontsize=12)

    
#Surrogate Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from io import StringIO
from pydot import graph_from_dot_data
from IPython.display import Image 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
y_pred_trainVal,_ = queue_pred(X_trainVal, gbm_reg, pred_length=48, window_size = W_size)

# Linear Regression surrogate
sg_model_linear = Ridge(alpha=100)
sg_model_linear = sg_model_linear.fit(X_trainVal, y_pred_trainVal)
sg_model_linear_mse = mean_squared_error(y_pred_trainVal, sg_model_linear.predict(X_trainVal))
sg_fi = sg_model_linear.coef_

w_list = sg_fi
feature_names = [X_train.columns[i].replace("_0103", "") for i in range(len(w_list))]
feature_names[10] = "MVART"
feature_names[8] = "Outdoor Humid. (1h MVA)"
feature_names[9] = "Outdoor Temp. (1h MVA)"
weight = [sg_fi[i] for i in range(len(w_list))]
abs_weight = list(map(lambda x:abs(x), weight))
name, abs_weight, weight = zip(*sorted(zip(feature_names, abs_weight, weight),key=lambda x: x[1], reverse=False))

print("*"*50)
print("MSE of the surrogate model: {:.6f}".format(sg_model_linear_mse))
print("Surrogate Model Feature Coefficients':")
for i in range(len(name)):
    print("{}: {:.8f}".format(name[len(name)-i-1], weight[len(name)-i-1]))

plt.figure()
plt.title("Feature Weights in Ridge Regression",fontsize=16)
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
# plt.barh(name,weight,color="royalblue")
plt.grid(True, axis="x")
plt.xlabel("Feature Coefficients")
plt.ylabel("Feature Name")
for i in range(len(weight)):
    if weight[i]>0:
        plt.text(weight[i]+0.05, name[i], "%.3f" % weight[i], ha='center', fontsize=11)
        plt.barh(name,weight,color="royalblue")
    else:
        plt.text(weight[i]-0.05, name[i], "%.3f" % weight[i], ha='center', fontsize=11)
        plt.barh(name,weight,color="orange")
        
lr_sg_coef = sg_model_linear.coef_
lr_sg_intercept = sg_model_linear.intercept_
print("The surrogate linear regression model:")
print_eq(lr_sg_coef, lr_sg_intercept, feature_names)
# Tree surrogate
sg_model_tree = DecisionTreeRegressor(criterion = "mse", max_depth=6, min_samples_split=0.02, min_samples_leaf=100)
sg_model_tree = sg_model_tree.fit(X_trainVal, y_pred_trainVal)
sg_model_tree_mse = mean_squared_error(y_pred_trainVal, sg_model_tree.predict(X_trainVal))
sg_fi = sg_model_tree.feature_importances_

w_list = sg_fi
weight = [sg_fi[i] for i in range(len(w_list))]
abs_weight = list(map(lambda x:abs(x), weight))
name, abs_weight, weight = zip(*sorted(zip(feature_names, abs_weight, weight),key=lambda x: x[1], reverse=False))

print("*"*50)
print("MSE of the surrogate model: {:.6f}".format(sg_model_tree_mse))
print("Surrogate Model Feature Importance:")
for i in range(len(name)):
    print("{}: {:.8f}".format(name[len(name)-i-1], weight[len(name)-i-1]))

plt.figure()
plt.title("Feature Importance for Surrogate Decision Tree Model",fontsize=16)
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.barh(name,weight,color="royalblue")
plt.grid(True, axis="x")
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
for i in range(len(weight)):
    plt.text(weight[i]+0.05, name[i], "%.4f" % weight[i], ha='center', fontsize=12)

# Save the Tree structure in PDF
# dot_data = StringIO()
# export_graphviz(sg_model_tree, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = feature_names)
# (graph, ) = graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

# graph.write_pdf("surrogate_tree_XGBM.pdf") 

# Frequency Analysis - on TrainVal Set
X_trainVal_hist = X_trainVal.copy()
for col in range(10):
    X_trainVal_hist.iloc[:,col] = X_trainVal_hist.iloc[:,col].mean()
pred_trainVal_hist,_ =  queue_pred(X_trainVal_hist, gbm_reg, pred_length=48, window_size = W_size)

trainVal_date = pd.Series((X_trainVal.index))
plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.plot(trainVal_date, pred_trainVal_hist,  label='Prediction')
plt.plot(trainVal_date, y_trainVal,  label='Ground True')
plt.xlim(trainVal_date.iloc[0],trainVal_date.iloc[-1])
plt.xlabel("Time")
plt.ylabel("SPT (Degree)")
plt.legend()
plt.title("XGBM Regressor Predictions v.s Ground True (Only Hist Features Valid)")
plt.show()

X_trainVal_other = X_trainVal.copy()
for col in range(10,X_trainVal_other.shape[1]):
    X_trainVal_other.iloc[:,col] = X_trainVal_other.iloc[:,col].mean()
pred_trainVal_other =  gbm_reg.predict(X_trainVal_other)

plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.plot(trainVal_date, pred_trainVal_other,  label='Prediction')
plt.plot(trainVal_date, y_trainVal,  label='Ground True')
#plt.ylim(23,40)
plt.xlim(trainVal_date.iloc[0],trainVal_date.iloc[-1])
plt.xlabel("Time")
plt.ylabel("SPT (Degree)")
plt.legend()
plt.title("XGBM Regressor Predictions v.s Ground True (Only Other Features Valid)")
plt.show()

# Frequency Analysis
    
oneSide_other_amp, oneSide_other_angle, oneside_frq_range = FFT_spectrum(pred_trainVal_other)
oneSide_hist_amp, oneSide_hist_angle, oneside_frq_range = FFT_spectrum(pred_trainVal_hist)
oneSide_y_test_amp, oneSide_y_test_angle, oneside_frq_range = FFT_spectrum(y_trainVal)
oneSide_y_test_pred_amp, oneSide_y_test_pred_angle, oneside_frq_range = FFT_spectrum(y_pred_trainVal)

end_f = 700
fig = plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
ax1 = fig.add_subplot(211)
ax1.plot(oneside_frq_range[1:end_f], oneSide_other_amp[1:end_f], label = "Other Features (DC: {:.2f})".format(oneSide_other_amp[0]))
ax1.plot(oneside_frq_range[1:end_f], oneSide_hist_amp[1:end_f], label = "Historical Features (DC: {:.2f})".format(oneSide_hist_amp[0]))
ax1.legend()
ax1.set_xlim(0, 2.5e-5)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Magnitude")
ax1.set_title("Frequency Responses for MVART Feature and Others (XGBM)")
ax2 = fig.add_subplot(212)
ax2.plot(oneside_frq_range[1:end_f], oneSide_y_test_amp[1:end_f], label = "True SPT (DC: {:.2f})".format(oneSide_y_test_amp[0]), c="g")
ax2.legend()
ax2.set_xlim(0, 2.5e-5)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_title("Frequency Responses for the True Values")

# PDP
from sklearn.inspection import plot_partial_dependence
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
gbm_reg.dummy_ = "dummy"
pdp_features = [i for i in range(11)]
pdp_features.remove(6)
for f in pdp_features:
    plot_partial_dependence(gbm_reg, X_trainVal, [f])

#%% Local Interpretation
deviate_samples = []
deviate_samples_values = []
accurate_samples = []
accurate_samples_values=[]
for i in range(X_val.shape[0]):
    if abs(y_pred_test[i]-y_test[i])>2 and y_test[i] not in deviate_samples_values:
        deviate_samples_values.append(y_test[i])
        deviate_samples.append(i)
    if abs(y_pred_test[i]-y_test[i])<0.01 and y_test[i] not in accurate_samples_values:
        accurate_samples_values.append(y_test[i])
        accurate_samples.append(i)    
local_samples = random.sample(deviate_samples,3)+random.sample(accurate_samples,3)
local_samples = [3047, 7241]
# LIME
local_explianer(model=gbm_reg, X_train=X_trainVal, X_test=X_test, y_test=y_test, random=False, sample_indicies= local_samples)

# SHAP
import shap

explainer = shap.TreeExplainer(gbm_reg)
shap_values = explainer.shap_values(X_test_)
plt.figure()
X_test_.columns=feature_names
shap.summary_plot(shap_values, X_test_,  plot_size=(12, 12))

for sample in local_samples:
    shap.force_plot(explainer.expected_value, shap_values[sample,:], X_test_.iloc[sample,:], matplotlib=True)
