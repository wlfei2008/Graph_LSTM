import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import sys
import os
import pandas as pd
from decimal import Decimal
import seaborn as sns
import matplotlib.font_manager
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv, find_dotenv
import multiprocessing
from scipy.interpolate import interp1d

class OutilerDete_Com:
    def __init__(self):
        self.inOutVecDim = 273  # number of stations
        file_dataset = '/home/longfei/Graph_LSTM_All/dataset/solar_data.csv'
        with open(file_dataset) as f:
            data = csv.reader(f, delimiter=",")
            solar = []
            for line in data:
                solar.append((line))
        
        self.solar = (np.array(solar)).astype(float) # all data
        self.solar = self.solar[:,:self.inOutVecDim]
        self.solar = self.Quartile_Outliers(self.solar, self.inOutVecDim)
        self.solar = self.OneclassSVM_Time(self.solar, self.inOutVecDim)
        self.solar = self.IsolationForest_Spatio(self.solar, self.inOutVecDim)
        # self.solar = self.sals_run(self.solar, 'sas', 15, 1.5, 2, 200, 5, 40)
        
    def reject_outliers(self, ser, q1, q2, q3, iqr): # Quartile_Outliers
        for i in range(len(ser)):
            if ser[i] > q3 + 1.5 * iqr:
                # print("Exced at:", i, ser[i])
                ser[i] = q3 + 1.5 * iqr
                # print("Exced at:", i, ser[i], q1, q2, q3)
            elif ser[i] < max(q2 - 1.5 * iqr , 0):
                ser[i] = max(q2 - 1.5 * iqr , 0)
                # print("Less at:", i)
        return ser
    
    def Quartile_Outliers(self, solar, inOutVecDim): # Detect and modify the values for each station using Quartile_Outliers
        print('...... Quartile_Outliers  ...')
        for i in range(inOutVecDim):
            # print(i)
            ser = pd.Series(solar[:,i])
            q1, q2, q3 = ser.quantile([0.25,0.5,0.75])
            iqr = q3 - q1
            ser = self.reject_outliers(ser, q1, q2, q3, iqr)
            solar[:,i] = ser.tolist()
        return solar 
        
    # # Unsupervised Outlier Detection using OneclassSVM and Isolation Forest algorithms for spatio-termal series data
    # def IsolationForest_Spatio(self, solar, Stat_ab_spatio):
        # print('...... Spatio TESTING  ...')
        # print(len(solar))
        # for j in range(10):
        # # Unsupervised Outlier Detection using Isolation Forests for spatio series time of each station
            # model_spatio = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12), \
                                           # max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
            # # model_spatio = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01) 
            # model_spatio.fit(solar[j,:].reshape(-1,1))
            # Stat_ab_spatio.append(model_spatio.predict(solar[j,:].reshape(-1, 1))) # predict results
            # print(model_spatio.predict(solar[j,:].reshape(-1, 1)))
        # return Stat_ab_spatio
        
    # Unsupervised Outlier Detection using OneclassSVM and Isolation Forest algorithms for spatio-termal series data
    def OneclassSVM_Time(self, solar, inOutVecDim):
        print('...... OneclassSVM_Time TESTING  ...')
        outliers_fraction=0.2 # an upper bound on the fraction of training errors
        nonzero_fraction=0.75 # an upper bound on the percent of nonzero data in each time step
        # Stat_ab_spatio = []
        # Stat_ab_spatio = self.IsolationForest_Spatio(solar, Stat_ab_spatio)
        for i in range(inOutVecDim):
            print('...... Time TESTING  ...')
            print(i)
            # oneclassSVM model for time series PV data classification
            model_time = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01) 
            model_time.fit(solar[:,i].reshape(-1, 1)) # oneclassSVM model fitting
            Stat_ab = pd.Series(model_time.predict(solar[:,i].reshape(-1, 1))) # predict results
            # print(Stat_ab)
            # To check each time step j data for station i is abnormal
            for j in range(1,len(solar)-1):
                if (Stat_ab[j] == -1) & (solar[j,i]==0.0): # complementate the abnormal zero data
                    # s = 'The time point is: ' + repr(j) + ', and the value is: ' + repr(solar[j,i]) + '...'
                    # print(s)
                    # print(np.count_nonzero(solar[j,:]))
                    if (solar[j-1,i]!=0.0) and (solar[j+1,i]!=0.0):
                        # s1 = 'The value is revised to: ' + repr((solar[j-1,i]+solar[j,i])/2) + '...Condition 1'
                        # print(s1)
                        solar[j,i] = (solar[j-1,i]+solar[j,i])/2
                    else:
                        if (np.count_nonzero(solar[j,:]) > nonzero_fraction*inOutVecDim):
                            solar[j,i] = np.nanmean(solar[j,:])
                            # s1 = 'The value is revised to: ' + repr(np.nanmean(solar[j,:])) + '...Condition 2'
                            # print(s1)
                        # else:
                            # s1 = 'The value is unchanged'
                            # print(s1)
                    # if (Stat_ab_spatio[j][i] == -1):
                        # mean_value = 0
                        # number_normal = 0
                        # for k in range(inOutVecDim):
                            # if (Stat_ab_spatio[j][i] != -1) & (k != i):
                                # mean_value += solar[j,k]
                                # number_normal += 1
                        # if number_normal > 0: mean_value /= number_normal
                        # s1 = 'The value is revised to: ' + repr(mean_value) + '...'
                        # print(s1)    
                        # solar[j,i] = mean_value
        return solar
    
    # Unsupervised Outlier Detection using OneclassSVM and Isolation Forest algorithms for spatio-termal series data
    def IsolationForest_Spatio(self, solar, inOutVecDim):
        print('...... IsolationForest_Spatio TESTING  ...')
        nonzero_fraction=0.75 # an upper bound on the percent of nonzero data in each time step
        # print('...... Spatio TESTING  ...')
        # print(len(solar))
        for j in range(len(solar)):
            print('...... Spatio TESTING  ...')
            print(j)
        # Unsupervised Outlier Detection using Isolation Forests for spatio series time of each station
            model_spatio = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12), \
                                           max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
            model_spatio.fit(solar[j,:].reshape(-1,1))
            Stat_ab_spatio = pd.Series(model_spatio.predict(solar[j,:].reshape(-1, 1))) # predict results
            for i in range(1,inOutVecDim-1):
                if (Stat_ab_spatio[i] == -1) & (solar[j,i]==0.0): # complementate the abnormal zero data
                    # s = 'The station is: ' + repr(i) + ', and the value is: ' + repr(solar[j,i]) + '...'
                    # print(s)
                    # print(np.count_nonzero(solar[j,:]))
                    if (solar[j-1,i]!=0.0) and (solar[j+1,i]!=0.0):
                        # s1 = 'The value is revised to: ' + repr((solar[j-1,i]+solar[j,i])/2) + '...Condition 1'
                        # print(s1)
                        solar[j,i] = (solar[j-1,i]+solar[j,i])/2
                    else:
                        if (np.count_nonzero(solar[j,:]) > nonzero_fraction*inOutVecDim):
                            solar[j,i] = np.nanmean(solar[j,:])
                            # s1 = 'The value is revised to: ' + repr(np.nanmean(solar[j,:])) + '...Condition 2'
                            # print(s1)
                        # else:
                            # s1 = 'The value is unchanged'
                            # print(s1)
        return solar

    def run(self):
        print('...... TESTING  ...')
        np.savetxt("/home/longfei/Graph_LSTM_All/dataset/solar_data_preprocessed.csv", self.solar, delimiter=",", fmt='%1.12f')

Preproce = OutilerDete_Com()
Preproce.run()