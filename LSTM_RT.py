# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 23:42:16 2021

@author: Mao Jianqiao
"""
#%% Configuration

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from datetime import datetime
import holidays
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import seaborn as sns

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

def lstm_data_form(x, y, num_steps, n_pred):
    # split into groups of num_steps
    X = np.array([x[i: i + num_steps]
                  for i in range(y.shape[0] - num_steps - n_pred +1)])
    y = np.array([y[i + num_steps: i + num_steps + n_pred]
                  for i in range(y.shape[0] - num_steps - n_pred +1)])
    
    return X, y    

## function to split the dataset by LSTM into training and testing sets
def lstm_data_split(x, y, num_steps, n_pred, train_size):

    # split into groups of num_steps
    X, y = lstm_data_form(x, y, num_steps, n_pred)

    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    
    return train_X, train_y, test_X, test_y

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
        print("Cosine Similarity: %.6f" %(r2))
        print("-"*50) 
        return mse, mae, mape, r2
    
def lstm_model(X, y, timestep, n_pred, batch_size, epoch, train_size, save=False, save_name=''):
    
    X_train, y_train, X_val, y_val = lstm_data_split(X, y, timestep, n_pred, train_size)
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(timestep, X_train.shape[2]),return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(n_pred, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()
    
    def scheduler(epoch):
        if epoch % 3 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(model.optimizer.lr)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=2, verbose=0, mode='min',
                                   baseline=None, restore_best_weights=True)

    reduce_lr = LearningRateScheduler(scheduler)
    
    if X_val.shape[0] != 0:
    
        model.fit(X_train, y_train, 
                  batch_size=batch_size, epochs=epoch, 
                  validation_data = (X_val, y_val), 
                  callbacks=[reduce_lr,early_stopping])
    else:
         model.fit(X_train, y_train, 
                   batch_size=batch_size, epochs=epoch)       
    
    preds = model.predict(np.vstack((X_train,X_val)))
    preds_train = preds[:len(y_train)]
    preds_val = preds[len(y_train):]
    
    if save:
        model.save("./"+save_name+".h5")
    
    return preds, preds_train, preds_val, model

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
            MVA_record = [x[i*pred_length,:,-1]]
            for j in range(pred_length-1):
                pred_j = model.predict(np.array(x[i*pred_length+j,:,:]).reshape(1,window_size,x_input.shape[2]), verbose=0)[0]
                y_pred.append(np.round(pred_j[0]*2)/2)
                MVA_record.append(pred_j[0])
                MVA = np.mean(MVA_record[-window_size:])
                x[i*pred_length+j+1,:,-1] = MVA
            pred_j = model.predict(np.array(x[(i+1)*pred_length-1,:,:]).reshape(1,window_size,x_input.shape[2]), verbose=0)[0]
            y_pred.append(np.round(pred_j[0]*2)/2)    
        else:
            MVA_record = [x[i*pred_length,:,-1]]
            for j in range(x.shape[0]-pred_length*int(x.shape[0]/pred_length)-1):
                pred_j = model.predict(np.array(x[i*pred_length+j,:,:]).reshape(1,window_size,x_input.shape[2]), verbose=0)[0]
                y_pred.append(np.round(pred_j[0]*2)/2)
                MVA_record.append(pred_j[0])
                MVA = np.mean(MVA_record[-window_size:])               
                x[i*pred_length+j+1,:,-1] = MVA
            if x.shape[0]%pred_length != 0:
                pred_j = model.predict(np.array(x[i*pred_length+x.shape[0]-pred_length*int(x.shape[0]/pred_length)-1,:,:]).reshape(1,window_size,x_input.shape[2]), verbose=0)[0]
                y_pred.append(np.round(pred_j[0]*2)/2)
    y_pred = np.array(y_pred).reshape(-1)     

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
features.loc[:,"OUTAIRHUMD"] = move_avg(features.loc[:,"OUTAIRHUMD"], W_size)
features.loc[:,"OUTAIRTEMP"] = move_avg(features.loc[:,"OUTAIRTEMP"], W_size)
MVA_RT_2h = move_avg(roomTemp, W_size)
features.loc[:,"1h_MVA"] = np.array([np.nan]+MVA_RT_2h)[:-1]

features=features.dropna()
roomTemp = roomTemp.iloc[1:]


#%% Dataset Spliting and Normalisation
X_train = features.loc[:split_point["Train_end"]]
y_train = roomTemp.loc[:split_point["Train_end"]]

X_val = features.loc[split_point["Val_begin"]:split_point["Val_end"]]
y_val = roomTemp.loc[split_point["Val_begin"]:split_point["Val_end"]]

X_test = features.loc[split_point["Test_begin"]:split_point["Test_end"]]
y_test = roomTemp.loc[split_point["Test_begin"]:split_point["Test_end"]]

norm_x = MinMaxScaler()
X_train.iloc[:,:] = norm_x.fit_transform(X_train.iloc[:,:])
X_val.iloc[:,:] = norm_x.transform(X_val.iloc[:,:])
X_test.iloc[:,:] = norm_x.transform(X_test.iloc[:,:])

norm_y =  MinMaxScaler()
y_train = norm_y.fit_transform(np.array(y_train.iloc[:]).reshape(-1,1))
y_val = norm_y.transform(np.array(y_val.iloc[:]).reshape(-1,1))
y_test = norm_y.transform(np.array(y_test.iloc[:]).reshape(-1,1))

X_trainVal = np.array(pd.concat([X_train, X_val]))
y_trainVal = np.vstack((y_train, y_val))

X_all = np.vstack((X_trainVal, X_test))
y_all = np.vstack((y_trainVal, y_test))
#%% Configure the LSTM
timestep = 6
n_pred = 48
batch_size = 4
epoch = 1
train_size = features.loc[:split_point["Train_end"]].shape[0]-timestep-n_pred+1

#%% Train

preds_trainVal, preds_train, preds_val, model = lstm_model(X_trainVal, y_trainVal, timestep, n_pred, batch_size, epoch, train_size, save=False, save_name='')
# model.save("./lstm_reg.h5")
# from tensorflow.keras.models import load_model
# model = load_model('./hist_models/lstms/reg/lstm_1.h5')
# print(model.summary())

#%% Validation

X_train, y_train, X_val, y_val = lstm_data_split(X_trainVal, y_trainVal, timestep, n_pred, train_size)
X_trainVal, y_trainVal = lstm_data_form(X_trainVal, y_trainVal, timestep, n_pred)
preds_trainVal = model.predict(X_trainVal)
preds_train = preds_trainVal[:len(y_train)]
preds_val = preds_trainVal[len(y_train):]

pred_trainVal_date = pd.Series((features.loc[:split_point["Val_end"]].index)[timestep:])

preds_trainVal_plot = np.array(list(preds_trainVal[0,:])+list(preds_trainVal[1:,-1]))
preds_trainVal_plot = norm_y.inverse_transform(preds_trainVal_plot.reshape(-1,1))

y_trainVal_plot = np.vstack((y_train, y_val))
y_trainVal_plot = np.array(list(y_trainVal_plot[0,:])+list(y_trainVal_plot[1:,-1]))
y_trainVal_plot = norm_y.inverse_transform(y_trainVal_plot.reshape(-1,1))

print("*"*75)
print("Training Phase:")
_=evaluate(norm_y.inverse_transform(preds_train),
           norm_y.inverse_transform(y_train.reshape(-1,n_pred)), 
           mode="reg")
print("*"*75)
print("Validation Phase:")
_=evaluate(norm_y.inverse_transform(preds_val),
           norm_y.inverse_transform(y_val.reshape(-1,n_pred)), 
           mode="reg")

## prediction plots
plt.figure()
plt.plot(pred_trainVal_date,y_trainVal_plot,  label='Ground True')
plt.plot(pred_trainVal_date, preds_trainVal_plot,  label='Prediction')
plt.ylim(13,40)
plt.xlim(pred_trainVal_date.iloc[0],pred_trainVal_date.iloc[-1])
plt.vlines(pred_trainVal_date[train_size-1],5,42,color="black",linestyle='--')
plt.fill_between(pred_trainVal_date[train_size-1:],5,42,facecolor='blue', alpha=0.3)
plt.legend()
plt.title("Predictions v.s Ground True (Train&Val Sets)")
plt.show()

#%% Test

X_test, y_test = lstm_data_form(np.array(X_test), y_test, timestep, n_pred)

preds_test = model.predict(X_test)

pred_test_date = pd.Series((features.loc[split_point["Test_begin"]:split_point["Test_end"]].index)[timestep:])

preds_test_plot = np.array(list(preds_test[0,:])+list(preds_test[1:,-1]))
preds_test_plot = norm_y.inverse_transform(preds_test_plot.reshape(-1,1))

y_test_plot = np.array(list(y_test[0,:])+list(y_test[1:,-1]))
y_test_plot = norm_y.inverse_transform(y_test_plot.reshape(-1,1))

print("*"*75)
print("Test Phase:")
_ = evaluate(norm_y.inverse_transform(preds_test),
             norm_y.inverse_transform(y_test.reshape(-1,n_pred)), 
             mode="reg")

plt.figure()
plt.plot(pred_test_date,y_test_plot,  label='Ground True')
plt.plot(pred_test_date, preds_test_plot,  label='Prediction')
plt.ylim(13,33)
plt.xlim(pred_test_date.iloc[0],pred_test_date.iloc[-1])
plt.legend()
plt.title("Predictions v.s Ground True (Test Set)")
plt.show()

#%% Overall predictions demo

X_all, y_all = lstm_data_form(X_all, y_all, timestep, n_pred)
preds_all = model.predict(X_all)

pred_all_date = pd.Series((features.loc[:split_point["Test_end"]].index)[timestep:])

preds_all_plot = np.array(list(preds_all[0,:])+list(preds_all[1:,-1]))
preds_all_plot = norm_y.inverse_transform(preds_all_plot.reshape(-1,1))

y_all_plot = np.array(list(y_all[0,:])+list(y_all[1:,-1]))
y_all_plot = norm_y.inverse_transform(y_all_plot.reshape(-1,1))

plt.figure()
plt.plot(pred_all_date,y_all_plot,  label='Ground True')
plt.plot(pred_all_date, preds_all_plot,  label='Prediction')
plt.ylim(13,40)
plt.xlim(pred_all_date.iloc[0],pred_all_date.iloc[-1])
plt.vlines(pred_all_date[train_size-1],0,40,color="black",linestyle='--')
plt.fill_between(pred_all_date[train_size-1:X_trainVal.shape[0]-2],0,40,facecolor='blue', alpha=0.3)
plt.vlines(pred_all_date[X_trainVal.shape[0]-1],0,40,color="black",linestyle='--')
plt.fill_between(pred_all_date[X_trainVal.shape[0]-1:],0,40,facecolor='red', alpha=0.3)
plt.legend()
plt.title("Predictions v.s Ground True")
plt.show()










