# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:03:10 2021

@author: Mao Jianqiao
"""
#%% Configuration

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller  
import holidays

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

def print_eq(coef, intercept):
    eq="y="
    for i in range(len(coef)):
        if coef[i]>=0:            
            eq += "{:.3f}*x{}".format(coef[i], i+1) 
        else:
            eq += "({:.3f})*x{}".format(coef[i], i+1) 
        if i!=len(coef-1):
            eq += "+"
    eq += "{:.3f}".format(intercept)   
    print(eq)
        

# R_res, R_unres, df    
def chow_test(X, y, break_point, alpha = 0.05):
    sub_lr_models = []
    feature_names = X.columns
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
        print("Period: {}-{}".format(X_train.index[0], X_train.index[-1]))
        print_eq(sub_coef, sub_intercept, feature_names)
        print("Sum of Square Residual (SSR): {:.2f}".format(R_unres))
        
    
    global_lr_model = LinearRegression()
    global_lr_model.fit(X,y)
    preds = global_lr_model.predict(X)
    
    global_coef = global_lr_model.coef_
    global_intercept = global_lr_model.intercept_
    R_res = sum((preds-y)**2)
    
    print("*"*50)
    print("The global linear regression model")
    print_eq(global_coef, global_intercept, feature_names)
    print("Sum of Square Residual (SSR): {:.2f}".format(R_res))
    print("*"*50)
    
    F = ((R_res-R_unres)/k)/(R_unres/df)
    p_value = stats.f.cdf(F, k, df)
    if p_value < alpha:
        print("There are structual changes at the selected time points with confidence_interval={}".format(1-alpha))
    else:
        print("There are not structual changes at the selected time points with confidence_interval={}".format(1-alpha))
    
    return F, [k,df], p_value, sub_lr_models, global_lr_model

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
#%% Feature Engineering

dataset = pd.read_csv("./Complete_Dataset.csv", usecols=[i+1 for i in range(7)])
dataset.loc[:,'Time'] = pd.to_datetime(dataset['Time'])
dataset = dataset.set_index('Time', drop=True)

roomTemp = dataset.loc[:,"RoomTemp_0103"]
features = dataset.drop(["RoomTemp_0103"], axis=1)

split_point = {"Train_begin": datetime.strptime("2017-12-08 15:39:36", '%Y-%m-%d %H:%M:%S'),
               "Train_end": datetime.strptime("2019-06-30 23:57:55", '%Y-%m-%d %H:%M:%S'),
               "Val_begin": datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'),
               "Val_end":  datetime.strptime("2019-10-10 23:56:41", '%Y-%m-%d %H:%M:%S'),
               "Test_begin": datetime.strptime("2019-10-11 00:06:41", '%Y-%m-%d %H:%M:%S'),
               "Test_end": datetime.strptime("2020-02-29 23:58:59", '%Y-%m-%d %H:%M:%S')}

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
MVA_RT_2h = move_avg(roomTemp, W_size)
features.loc[:,"RT_1h_HistoricalMVA"] = np.array([np.nan]+MVA_RT_2h)[:-1]

features=features.dropna()
roomTemp = roomTemp.iloc[1:]

#%% EDA

dataset_EDA = features.copy()
dataset_EDA = dataset_EDA.drop(["WeekDay","Hour","Month","Quarter"], axis=1)
feature_names = dataset_EDA.columns

## Correlation matrix
# Continous Variables
cont_ds = dataset_EDA.loc[:,["OUTAIRHUMD","OUTAIRTEMP","SetTemp_0103"]]
corr_coef = cont_ds.corr(method='spearman')
sticks = ["Outside Humd.", "Outside Temp.", "SPT"]
fig = plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
sns.heatmap(corr_coef, cmap=plt.cm.viridis,annot=True, xticklabels=sticks, yticklabels=sticks)
plt.title("Feature Correlation for Continuous Variables")

# Categorical Variables
cat_variables = ["OnOffState_0103", "OperationModeState_0103", "OOS_OC", "Holiday?"]
cat_feature_names = ["On/Off State", "Operation Mode State", "Occupancy State", "Holiday"]
cat_data = dataset_EDA.loc[:,cat_variables]
comb_list = list(itertools.combinations(cat_variables , 2))
comb_sticks = list(itertools.combinations(cat_feature_names , 2))
cross_tables = []
p_list = [] 
for comb in range(len(comb_list)):
    cross_table = pd.crosstab(cat_data[comb_list[comb][0]],cat_data[comb_list[comb][1]])
    
    plt.figure(facecolor='lightgray')
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16  
    plt.rcParams["xtick.labelsize"] = "large"
    plt.rcParams["ytick.labelsize"] = "large"
    plt.title(comb_sticks[comb][0]+" v.s "+comb_sticks[comb][1], fontsize=16)
    ax=sns.heatmap(cross_table, fmt='d',cmap='Greys',annot=True)
    ax.set_xlabel(comb_sticks[comb][1])
    ax.set_ylabel(comb_sticks[comb][0])
    plt.show()     
    
    cross_tables.append(cross_table)
    chi2, p, dof, expected = chi2_contingency(cross_table)
    p_list.append(p)
    if p<0.05:
        print("Categorical variables: {} and {} are CORRELATED under confidence interval of 0.99.".format(comb_list[comb][0], comb_list[comb][1]))
    else:
        print("Categorical variables: {} and {} are NOT CORRELATED under confidence interval of 0.99.".format(comb_list[comb][0], comb_list[comb][1]))

## Visualisation
dataset_EDA.loc[:,"roomTemp"] = roomTemp
holiday_date = list(set(list(dataset_EDA.loc[dataset_EDA.loc[:,"Holiday?"]==1,:].index.date)))

# Continous Features
fig = plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 18  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
ax1 = fig.add_subplot(311)
ax1.plot(dataset_EDA.loc[:split_point["Test_end"], "roomTemp"], label="Room Tempeature")
# ax1.plot(dataset_EDA.index,MVA_RT, label="1-hour Moving Average")
ax1.vlines(split_point["Train_end"],13,40,color="black",linestyle='--')
ax1.vlines(split_point["Val_end"],13,40,color="black",linestyle='--')
ax1.set_ylim(13,40)
ax1.set_xlim(split_point["Train_begin"],split_point["Test_end"])
ax1.fill_between(dataset_EDA.loc[split_point["Val_begin"]:split_point["Val_end"]].index,13,40,facecolor='blue', alpha=0.3)
ax1.fill_between(dataset_EDA.loc[split_point["Test_begin"]:split_point["Test_end"]].index,13,40,facecolor='red', alpha=0.3)
ax1.set_xlabel("Time")
ax1.set_ylabel("Temp. (Celsius)")
ax1.legend()
ax1.set_title("Room Tempeature (Target)")

ax2 = fig.add_subplot(312)
ln1 = ax2.plot(dataset_EDA.loc[:split_point["Test_end"], "OUTAIRHUMD"], label="Outside Humidity", c="darkorange")
ax2.set_xlim(split_point["Train_begin"],split_point["Test_end"])
ax2.set_ylabel("Humidity (%)")
ax2.set_xlabel("Time")
ax2.fill_between(dataset_EDA.loc[split_point["Val_begin"]:split_point["Val_end"]].index,0,90,facecolor='blue', alpha=0.3)
ax2.fill_between(dataset_EDA.loc[split_point["Test_begin"]:split_point["Test_end"]].index,0,90,facecolor='red', alpha=0.3)
ax2.vlines(split_point["Train_end"],35,90,color="black",linestyle='--')
ax2.vlines(split_point["Val_end"],35,90,color="black",linestyle='--')
ax2.set_ylim(35,90)
ax2_multi = ax2.twinx()
ln2 = ax2_multi.plot(dataset_EDA.loc[:split_point["Test_end"], "OUTAIRTEMP"], label="Outside Temp.")
ln2 += ax2_multi.plot(dataset_EDA.loc[:split_point["Test_end"], "SetTemp_0103"], label="Set Point Temp.", c="g")
ax2_multi.set_ylabel("Temp. (Celsius)")
ax2_multi.set_ylim(0,40)
ln_mul = ln1+ln2
labels = [l.get_label() for l in ln_mul ]
ax2_multi.legend(ln_mul, labels, loc="lower right")
ax2.set_title("Continuous Features")

# Categorical Features
ax3 = fig.add_subplot(313)
count=1
OP_states = ["Cooling", "Heating", "Ventilation"]
dataset_EDA.loc[:,OP_states[0]] = dataset_EDA.loc[:,"OperationModeState_0103"]==1
dataset_EDA.loc[:,OP_states[0]] = dataset_EDA.loc[:,OP_states[0]].map(lambda x:int(x))
dataset_EDA.loc[:,OP_states[1]] = dataset_EDA.loc[:,"OperationModeState_0103"]==2
dataset_EDA.loc[:,OP_states[1]] = dataset_EDA.loc[:,OP_states[1]].map(lambda x:int(x))
dataset_EDA.loc[:,OP_states[2]] = dataset_EDA.loc[:,"OperationModeState_0103"]==3
dataset_EDA.loc[:,OP_states[2]] = dataset_EDA.loc[:,OP_states[2]].map(lambda x:int(x))
for state in OP_states:
    acting = dataset_EDA.loc[dataset_EDA.loc[:,state]==1,state]
    acting.iloc[:] = count
    count+=1
    ax3.plot(acting.index, acting.iloc[:], '|')
SysOn = dataset_EDA.loc[dataset_EDA.loc[:,"OnOffState_0103"]==1,"OnOffState_0103"]
SysOn.iloc[:]=4
ax3.plot(SysOn.index, SysOn.iloc[:], "|")
OC_state = dataset_EDA.loc[dataset_EDA.loc[:,"OOS_OC"]==1,"OOS_OC"]
OC_state.iloc[:]=5
ax3.plot(OC_state.index, OC_state.iloc[:], "|")
Holiday = dataset_EDA.loc[dataset_EDA.loc[:,"Holiday?"]==1,"Holiday?"]
Holiday.iloc[:]=6
ax3.plot(Holiday.index, Holiday.iloc[:], "|")
ax3.vlines(split_point["Train_end"],0.5,6.5,color="black",linestyle='--')
ax3.vlines(split_point["Val_end"],0.5,6.5,color="black",linestyle='--')
ax3.set_ylim(0.5,6.5)
ax3.set_xlim(split_point["Train_begin"],split_point["Test_end"])
ax3.fill_between(dataset_EDA.loc[split_point["Val_begin"]:split_point["Val_end"]].index,0.5,6.5,facecolor='blue', alpha=0.3)
ax3.fill_between(dataset_EDA.loc[split_point["Test_begin"]:split_point["Test_end"]].index,0.5,6.5,facecolor='red', alpha=0.3)
ax3.set_xlabel("Time")
ax3.set_ylabel("HVAC States")
plt.yticks(ticks=range(1,7),labels=["Cooling", "Heating", "Ventilation", "System On", "Occupancy", "Holiday"])
ax3.set_title("Categorical Features")

# Monthly Details to explore the holiday effects
fig = plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 18  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.plot(dataset_EDA.loc[datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'):datetime.strptime("2019-07-31 23:57:55", '%Y-%m-%d %H:%M:%S'), "roomTemp"], label="Room Tempeature")
plt.ylim(22,38)
plt.xlim(datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'),datetime.strptime("2019-07-31 23:57:55", '%Y-%m-%d %H:%M:%S'))
plt.xlabel("Time")
plt.ylabel("Temp. (Celsius)")
plt.title("Room Tempeature from 1st to 31st of July, 2019")
weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.xticks([dataset_EDA.loc[datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'):datetime.strptime("2019-07-31 23:57:55", '%Y-%m-%d %H:%M:%S'), "roomTemp"].index[i*144] for i in range(31)], [weekday[i%7] for i in range(31)])

# Moving Average Visualisation
fig = plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 18  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(dataset_EDA.loc[datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'):datetime.strptime("2019-07-31 23:57:55", '%Y-%m-%d %H:%M:%S'), "OUTAIRHUMD"], label="Outside Humidity" )
ln1 += ax1.plot(dataset_EDA.loc[datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'):datetime.strptime("2019-07-31 23:57:55", '%Y-%m-%d %H:%M:%S'), "OUTAIRHUMD_1h_MVA"], label="Outside Humidity (1-hour MVA)")
ax1.set_xlim(datetime.strptime("2019-07-02 00:07:55", '%Y-%m-%d %H:%M:%S'),datetime.strptime("2019-07-03 23:57:55", '%Y-%m-%d %H:%M:%S'))
ax1.set_ylabel("Humidity (%)")
ax1.set_xlabel("Time")
ax1.set_ylim(41,62)
ax1_multi = ax1.twinx()
ln2 = ax1_multi.plot(dataset_EDA.loc[datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'):datetime.strptime("2019-07-31 23:57:55", '%Y-%m-%d %H:%M:%S'), "OUTAIRTEMP"], label="Outside Temp.", c="r")
ln2 += ax1_multi.plot(dataset_EDA.loc[datetime.strptime("2019-07-01 00:07:55", '%Y-%m-%d %H:%M:%S'):datetime.strptime("2019-07-31 23:57:55", '%Y-%m-%d %H:%M:%S'), "OUTAIRTEMP_1h_MVA"], label="Outside Temp.(1-hour MVA)", c="g")
ax1_multi.set_ylabel("Temp. (Celsius)")
ax1_multi.set_ylim(26,34)
ln_mul = ln1+ln2
labels = [l.get_label() for l in ln_mul ]
ax1_multi.legend(ln_mul, labels, loc="lower right")
plt.title("Continuous Features With/Without MVA Filters")

## Target Distribution
dataset_EDA.loc[:,"Room Temp."] = np.round(dataset_EDA.loc[:,"roomTemp"]).map(lambda x:int(x))
fig = plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
ax1 = fig.add_subplot(221)    
sns.countplot(x="Room Temp.",data=dataset_EDA.loc[:], palette=["darkslateblue"])
ax1.set_title("Overall Dataset Distribution")

ax2 = fig.add_subplot(222)    
sns.countplot(x="Room Temp.",data=dataset_EDA.loc[:split_point["Train_end"]], palette=["darkslateblue"])
ax2.set_title("Training Set Distribution")

ax3 = fig.add_subplot(223)    
sns.countplot(x="Room Temp.",data=dataset_EDA.loc[split_point["Val_begin"]:split_point["Val_end"]], palette=["darkslateblue"])
ax3.set_title("Val Set Distribution")

ax4 = fig.add_subplot(224)    
sns.countplot(x="Room Temp.",data=dataset_EDA.loc[split_point["Test_begin"]:split_point["Test_end"]], palette=["darkslateblue"])
ax4.set_title("Test Set Distribution")

## Stationarity Test
# Autocorrelation and Partial Autocorrelation
fig = plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
axes1 = fig.add_subplot(211)
plot_acf(dataset_EDA.loc[:,"roomTemp"], ax = axes1, title='Room Temp. Autocorrelation')
axes1.set_xlim([-0.5,30.5])
axes1.set_xlabel('Correlation Order')
axes1.set_ylabel('Correlation Score')
axes2 = fig.add_subplot(212)
plot_pacf(dataset_EDA.loc[:,"roomTemp"], ax = axes2, title='Room Temp. Partial Autocorrelation')
axes2.set_xlim([-0.5,30.5])
axes2.set_xlabel('Correlation Order')
axes2.set_ylabel('Correlation Score')

# Autocorrelation and Partial Autocorrelation for 1st-order diff
fig = plt.figure()
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16  
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
axes1 = fig.add_subplot(211)
plot_acf(dataset_EDA.loc[:,"roomTemp"].diff().dropna(), ax = axes1, title='Room Temp. 1st-order Difference Autocorrelation')
axes1.set_xlim([-0.5,30.5])
axes1.set_xlabel('Correlation Order')
axes1.set_ylabel('Correlation Score')
axes2 = fig.add_subplot(212)
plot_pacf(dataset_EDA.loc[:,"roomTemp"].diff().dropna(), ax = axes2, title='Room Temp. 1st-order Difference Partial Autocorrelation')
axes2.set_xlim([-0.5,30.5])
axes2.set_xlabel('Correlation Order')
axes2.set_ylabel('Correlation Score')


# Unit root hyperpothesis test using ADF
# H0: non-stationary has unit root
# H1: stationary
RT_stationarity_test = adfuller(x=dataset_EDA.loc[:,"roomTemp"], regression = 'nc')
p_stationarity = RT_stationarity_test[1]
if p_stationarity<0.01: # Reject the Null
    print("Set Point Temp. is a stationary series under the confidence interval 0.99")
else:
    print("Set Point Temp. is a non-stationary series under the confidence interval 0.99")

RT_dif_stationarity_test = adfuller(x=dataset_EDA.loc[:,"roomTemp"].diff().dropna(), regression = 'c')
p_stationarity_dif = RT_dif_stationarity_test[1]
if p_stationarity_dif<0.01: # Reject the Null
    print("Set Point Temp. is a stationary series under the confidence interval 0.99")
else:
    print("Set Point Temp. is a non-stationary series under the confidence interval 0.99")

# Features v.s. target
comb_list = list(itertools.combinations(feature_names[2:5] , 2))
for comb in range(len(comb_list)):
    x_range = (int(0.9*min(dataset_EDA.loc[:,comb_list[comb][0]])), int(1.1*max(dataset_EDA.loc[:,comb_list[comb][0]])))
    y_range = (int(0.9*min(dataset_EDA.loc[:,comb_list[comb][1]])), int(1.1*max(dataset_EDA.loc[:,comb_list[comb][1]])))
    
    plt.rcParams["axes.labelsize"] = 14 
    plt.rcParams["xtick.labelsize"] = "large"
    plt.rcParams["ytick.labelsize"] = "large"
    plt1=sns.jointplot(x=comb_list[comb][0], y=comb_list[comb][1], 
                       data=dataset_EDA.loc[:split_point["Train_end"]], 
                       kind='hist', hue="Room Temp.",
                       xlim=(x_range[0],x_range[1]), ylim=(y_range[0],y_range[1]))
    plt1.fig.suptitle("Training Set",fontsize=16)
    
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = "large"
    plt.rcParams["ytick.labelsize"] = "large"
    plt2=sns.jointplot(x=comb_list[comb][0], y=comb_list[comb][1], 
                       data=dataset_EDA.loc[split_point["Val_begin"]:split_point["Val_end"]], 
                       kind='hist', hue="Room Temp.",
                       xlim=(x_range[0],x_range[1]), ylim=(y_range[0],y_range[1]))
    plt2.fig.suptitle("Validation Set",fontsize=16)
    plt2.fig.tight_layout()

    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = "large"
    plt.rcParams["ytick.labelsize"] = "large" 
    plt3=sns.jointplot(x=comb_list[comb][0], y=comb_list[comb][1], 
                       data=dataset_EDA.loc[split_point["Test_begin"]:split_point["Test_end"]], 
                       kind='hist', hue="Room Temp.",
                       xlim=(x_range[0],x_range[1]), ylim=(y_range[0],y_range[1]))  
    plt3.fig.suptitle("Test Set",fontsize=16)
    plt3.fig.tight_layout()

# Chow-test
features_chow = ['OnOffState_0103', 'OperationModeState_0103', "SetTemp_0103", "OUTAIRTEMP",'OUTAIRHUMD', 'OOS_OC', "Holiday?"]
F, df, p_value, sub_lr_models, global_lr_model= chow_test(X=features.loc[:split_point["Test_end"]].loc[:,features_chow],
                                                      y=roomTemp[:split_point["Test_end"]],
                                                      break_point=[split_point["Train_end"],split_point["Val_end"]],
                                                      alpha = 0.01)
print("F_statistics: {:.2f} (Degree of Freedom: {}, {}), p-value:{:.4f}".format(F, df[0], df[1], p_value))




