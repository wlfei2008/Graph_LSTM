from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,Normalizer
from datetime import datetime, timedelta
import numpy as np
import pickle
import os
import datetime
import time
import matplotlib.pyplot as plt
import csv
import sys
from decimal import Decimal
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from dotenv import load_dotenv, find_dotenv
import multiprocessing
from scipy.interpolate import interp1d

# convert X_train file into solar file for multi-LSTM
Wea_inOutVecDim = 33  # number of parameters in weather data
file_dataset =  '/home/longfei/Graph_LSTM_All/dataset/W_train.csv'
with open(file_dataset, encoding="utf8") as f:
    data = csv.reader(f, delimiter=",")
    weather = []
    for line in data:
        weather.append((line))
weather = np.array(weather)
weather = weather[:,:Wea_inOutVecDim]  
weather_Data = pd.DataFrame(data=weather[1:,:], columns=weather[0,:])
indexNames = weather_Data[ weather_Data['areaid'] == '268' ].index # Get names of indexes for which column areaid has value 268
weather_Data.drop(indexNames , inplace=True) # Delete these row indexes from dataFrame since data missing for areaid 268
print("Select the specific weather parameters from W_trian")
weather_Data_Selected = weather_Data[['areaid','latitude','longitude','all_sky_sfc_sw_dwn','all_sky_sfc_lw_dwn']]
weather_Data_Selected = weather_Data_Selected.astype(float) # all data
weather_Data_mean = weather_Data_Selected.groupby('areaid').mean() # mean value for each area

# create kmeans object (3 clusters) for clustering 57 PV sites
kmeans_spatio = KMeans(n_clusters=3)
kmeans_spatio.fit(weather_Data_mean)
cluster_spatio = kmeans_spatio.fit_predict(weather_Data_mean) # save new clusters for chart
cluster1_site = [i for i, x in enumerate(cluster_spatio) if x == 0]
print(cluster1_site)
cluster2_site = [i for i, x in enumerate(cluster_spatio) if x == 1]
print(cluster2_site)
cluster3_site = [i for i, x in enumerate(cluster_spatio) if x == 2]
print(cluster3_site)
spatio_cluster = np.column_stack((list(range(1,274)), weather_Data_mean.index, cluster_spatio))
print("File save")
np.savetxt("/home/longfei/Graph_LSTM_All/dataset/spatio_cluster.csv", spatio_cluster, delimiter=",", fmt='%1f')

# load solar data
inOutVecDim = 273  # number of stations
file_dataset = '/home/longfei/Graph_LSTM_All/dataset/solar_data_final.csv'
with open(file_dataset) as f:  # open solar generation data 
    data = csv.reader(f, delimiter=",")
    solar = []
    for line in data:
        solar.append((line))
solar = (np.array(solar)).astype(float) # all data
solar = solar[:,:inOutVecDim]
solar_data = pd.DataFrame(solar)  # solar dataframe
print('...... TESTING  ...')
np.savetxt("/home/longfei/Graph_LSTM_All/dataset/solar_data_cluster1.csv", solar_data[cluster1_site], delimiter=",", fmt='%1.12f')
np.savetxt("/home/longfei/Graph_LSTM_All/dataset/solar_data_cluster2.csv", solar_data[cluster2_site], delimiter=",", fmt='%1.12f')
np.savetxt("/home/longfei/Graph_LSTM_All/dataset/solar_data_cluster3.csv", solar_data[cluster3_site], delimiter=",", fmt='%1.12f')