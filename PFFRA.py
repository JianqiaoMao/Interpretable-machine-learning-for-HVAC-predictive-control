# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 23:11:20 2023

@author: NickMao
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PF_FRA():
    
    def __init__(self, y, X, model, feature_name, sampling_intervals):
        self.y = y
        self.X = X       
        self.model = model
        self.feature_name = feature_name
        self.sampling_intervals = sampling_intervals
        
        
        # if self.preds_list == list:
        #     raise TypeError('preds_list accepts "list" input.')
            
    def permutation_pred(self):
        
        X_feature = self.X.copy()
        feature_names = list(self.X.columns)
        feature_names.remove(self.feature_name)
        for col in feature_names:
            X_feature.loc[:,col] = X_feature.loc[:,col].mean()
        pred_feature =  self.model.predict(X_feature)
        
        X_other = self.X.copy()
        X_other.loc[:,self.feature_name] = X_other.loc[:,self.feature_name].mean()
        pred_other = self.model.predict(X_other)
        
        return pred_feature, pred_other
        
    
    def FFT_spectrum(self, series):
        sampling_t = self.sampling_intervals
        F_samp = 1/sampling_t
        n = series.shape[0]
        k = np.arange(n)
        T = n/F_samp
        twoside_frq_range = k/T
        oneSide_frq_range  = twoside_frq_range[range(int(n/2))]     
        
        spectrum = np.fft.fft(series)/n
        twoSide_amp = np.abs(spectrum)
        oneSide_amp = twoSide_amp[range(int(n/2))]
        twoSide_angle = np.angle(spectrum)
        oneSide_angle = twoSide_angle[range(int(n/2))]
    
        return oneSide_amp, oneSide_angle, oneSide_frq_range

    def show(self, end_position, rename_feature='Interested Feature'):
    
        pred_feature, pred_other = self.permutation_pred()
        
        oneSide_other_amp, oneSide_other_angle, oneside_frq_range = self.FFT_spectrum(pred_other)
        oneSide_feature_amp, oneSide_feature_angle, oneside_feature_range = self.FFT_spectrum(pred_feature)
        oneSide_y_test_amp, oneSide_y_test_angle, oneside_frq_range = self.FFT_spectrum(self.y)
        
        end_f = end_position
        fig = plt.figure()
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["axes.titlesize"] = 16  
        plt.rcParams["xtick.labelsize"] = "large"
        plt.rcParams["ytick.labelsize"] = "large"
        ax1 = fig.add_subplot(211)
        ax1.plot(oneside_frq_range[1:end_f], oneSide_other_amp[1:end_f], label = "Other Features (DC: {:.2f})".format(oneSide_other_amp[0]))
        ax1.plot(oneside_frq_range[1:end_f], oneSide_feature_amp[1:end_f], label = "{} (DC: {:.2f})".format(rename_feature, oneSide_feature_amp[0]))
        ax1.legend()
        ax1.set_xlim(0, 2.5e-5)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude")
        ax1.set_title("Frequency Responses for the {} Feature and Others".format(rename_feature))
        ax2 = fig.add_subplot(212)
        ax2.plot(oneside_frq_range[1:end_f], oneSide_y_test_amp[1:end_f], label = "True SPT (DC: {:.2f})".format(oneSide_y_test_amp[0]), c="g")
        ax2.legend()
        ax2.set_xlim(0, 2.5e-5)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_title("Frequency Responses for the True Values")
        
# pffra_instance = PF_FRA(y = y_trainVal, X = X_trainVal, model = gbm_reg, feature_name="RT_1h_HistoricalMVA", sampling_intervals=300)
# pffra_instance.PF_FRA(end_position = 700)