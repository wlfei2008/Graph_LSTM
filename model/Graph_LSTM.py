import math
import pandas as pd
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,Normalizer
from datetime import datetime, timedelta
import numpy as np
import pickle
import os
import datetime
from decimal import Decimal
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from collections import Counter, defaultdict
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs
from dotenv import load_dotenv, find_dotenv
import multiprocessing
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import ELU
from keras.layers import LeakyReLU
import keras as keras
np.random.seed(1234)
from keras import backend as K
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

class multiLSTM(object):
    def __init__(self, i, cluster_site):
        self.inputHorizon = 59*4 # number of time steps as input
        # self.inOutVecDim = 17  # number of station
        self.elu_alpha = 0.1 # alpha for ELU layer
        self.llu_alpha = 0.01 # alpha for ELU layer
        self.lstmModels = [ None for _ in range(6)]
        self.xTest, self.yTest = None, None
        self.cluster = i
        file_dataset = '/home/longfei/Graph_LSTM_All/dataset/solar_data_final.csv'
        with open(file_dataset) as f:
            data = csv.reader(f, delimiter=",")
            solars = []
            for line in data:
                solars.append((line))
                
        self.solars = (np.array(solars)).astype(float) # all data
        print("Select the right cluster")
        print(cluster_site)
        self.solars = self.solars[:,cluster_site]
        print(self.solars.shape)
        self.inOutVecDim = self.solars.shape[1]
        # self.solars = self.solars[:,:self.inOutVecDim]
        self.means_stds = [0,0]
        self.solars, self.means_stds = self.normalize_solars_0_1(self.solars)
        self.validation_split = 0.05
        self.batchSize = 30
        activation = ['sigmoid',  "tanh",  "relu", 'linear']
        self.activation = activation[3]
        realRun = 1
        #          model number :           1   2   3   4   5   6
        self.epochs, self.trainDataRate = [[20, 20, 20, 20, 20, 20], 1] if realRun else [[ 1, 1, 1, 1, 1, 1] , 0.005]# percentage of data used for training(saving time for debuging)

    def normalize_solars_0_1(self, solars):
        '''normalize based on each station data'''
        stations = solars.shape[1]
        normal_solars = []
        mins_maxs = []
        solarMax = solars.max()
        solarMin = solars.min()
        normal_solars = (solars - solarMin) / solarMax
        mins_maxs = [solarMin, solarMax]
        return np.array(normal_solars), mins_maxs

    def denormalize(self, vec):
        res = vec * self.means_stds[1] + self.means_stds[0]        #  fro 0 to 1
        return res

    def loadData_1(self):
        # for lstm1 output xtrain ytrain
        result = []
        for index in range(len(self.solars) - self.inputHorizon):
            result.append(self.solars[index:index + self.inputHorizon])
        result = np.array(result)  

        # trainRow = int(15000 * self.trainDataRate)
        trainRow = int((self.solars.shape[0]-11*59) * self.trainDataRate)
        # trainRow = int(94*59 * self.trainDataRate)# 14,337 = 243 * 59 + 413 from 1/1 to 8/31 to 9/7
        X_train = result[:trainRow, :]
        y_train = self.solars[self.inputHorizon:trainRow + self.inputHorizon]
        self.xTest = result[(self.solars.shape[0]-11*59):(self.solars.shape[0]-11*59)+413, :] # 9/8 to 9/14
        self.yTest = self.solars[(self.solars.shape[0]-11*59) + self.inputHorizon:(self.solars.shape[0]-11*59)+413 + self.inputHorizon] # 9/12 to 9/18
        # self.xTest = result[14337:(14337+413), :] # from 9/1 to 9/7
        # self.yTest = self.solars[14337 + self.inputHorizon:(14337+413) + self.inputHorizon] # from 9/8 to 9/14
        self.predicted = np.zeros_like(self.yTest)
        return [X_train, y_train]

    def loadData(self, preXTrain, preYTrain, model): # xtrain and ytrain from loadData_1
        # for lstm2 output: xtrain ytrain
        xTrain, yTrain = np.ones_like(preXTrain), np.zeros_like(preYTrain)
  
        for ind in range(len(preXTrain) - self.inputHorizon -1):
            tempInput = preXTrain[ind]
            temp_shape = tempInput.shape
            tempInput = np.reshape(tempInput, (1,temp_shape[0],temp_shape[1]))
            output = model.predict(tempInput)
            tInput = np.reshape(tempInput,temp_shape)
            tempInput = np.vstack((tInput, output))
            tempInput = np.delete(tempInput, 0, axis=0)
            xTrain[ind] = tempInput
            yTrain[ind] = preYTrain[ind+1]
        return [xTrain, yTrain]

  
    def buildModelLSTM_1(self):
        model = Sequential()
        in_nodes = out_nodes = self.inOutVecDim
        layers = [in_nodes, 20*2, 20, 32, out_nodes]
        model.add(LSTM(input_dim=layers[0],output_dim=layers[1],
            return_sequences=False))
    
        model.add(Dense(
            output_dim=layers[4]))
        # model.add(Activation(self.activation))
        model.add(ELU(alpha = self.elu_alpha))
        # model.add(LeakyReLU(alpha = self.llu_alpha))
         
        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_2(self):
        model = Sequential()
        layers = [self.inOutVecDim, 10 , 20 * 2, 32, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0],output_dim=layers[1],
            return_sequences=False))

        model.add(Dense(
            output_dim=layers[4]))

        # model.add(Activation(self.activation))
        model.add(ELU(alpha = self.elu_alpha))
        # model.add(LeakyReLU(alpha = self.llu_alpha))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_3(self):
        model = Sequential()

        layers = [self.inOutVecDim, 20, 20 * 2, 32, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1],
            return_sequences=False))

        model.add(Dense(
            output_dim=layers[4]))

        # model.add(Activation(self.activation))
        model.add(ELU(alpha = self.elu_alpha))
        # model.add(LeakyReLU(alpha = self.llu_alpha))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_4(self):
        model = Sequential()

        layers = [self.inOutVecDim, 20, 20 * 2, 20, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1],
            return_sequences=True))

        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(output_dim=layers[4]))

        # model.add(Activation(self.activation))
        model.add(ELU(alpha = self.elu_alpha))
        # model.add(LeakyReLU(alpha = self.llu_alpha))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_5(self):
        model = Sequential()

        layers = [self.inOutVecDim, 30, 20 * 2, 20, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1],
            return_sequences=False))

        model.add(Dense(output_dim=layers[4]))

        # model.add(Activation(self.activation))
        model.add(ELU(alpha = self.elu_alpha))
        # model.add(LeakyReLU(alpha = self.llu_alpha))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_6(self):
        model = Sequential()
        layers = [self.inOutVecDim, 20*2, 20 * 2, 20, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1],
        return_sequences=True))


        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(output_dim=layers[4]))

        # model.add(Activation(self.activation))
        model.add(ELU(alpha = self.elu_alpha))
        # model.add(LeakyReLU(alpha = self.llu_alpha))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM(self, lstmModelNum):
        if   lstmModelNum == 1:
            return self.buildModelLSTM_1()
        elif lstmModelNum == 2:
            return self.buildModelLSTM_2()
        elif lstmModelNum == 3:
            return self.buildModelLSTM_3()
        elif lstmModelNum == 4:
            return self.buildModelLSTM_4()
        elif lstmModelNum == 5:
            return self.buildModelLSTM_5()
        elif lstmModelNum == 6:
            return self.buildModelLSTM_6()

    def trainLSTM(self, xTrain, yTrain, lstmModelNum):
        # train first LSTM with inputHorizon number of real input values

        lstmModel = self.buildModelLSTM(lstmModelNum)
        lstmModel.fit(xTrain, yTrain, batch_size=self.batchSize, nb_epoch=self.epochs[lstmModelNum-1], validation_split=self.validation_split)
        return lstmModel

    def test(self):
        ''' calculate the predicted values(self.predicted) '''
        for ind in range(len(self.xTest)):
            modelInd = ind % 6
            if modelInd == 0:
                testInputRaw = self.xTest[ind]
                testInputShape = testInputRaw.shape
                testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            else :
                testInputRaw = np.vstack((testInputRaw, self.predicted[ind-1]))
                testInput = np.delete(testInputRaw, 0, axis=0)
                testInputShape = testInput.shape
                testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])

            self.predicted[ind] = self.lstmModels[modelInd].predict(testInput)
            # code to replace all negative value with 0 
            self.predicted[self.predicted < 0] = 0

        return

    def errorMeasures(self, denormalYTest, denormalYPredicted):

        mae = np.mean(np.absolute(denormalYTest - denormalYPredicted))
        nmae = 100 * mae / (np.amax(denormalYTest) - np.amin(denormalYTest))
        rmse = np.sqrt((np.mean((np.absolute(denormalYTest - denormalYPredicted)) ** 2)))
        nrsme_maxMin = 100*rmse / (denormalYTest.max() - denormalYTest.min())
        nrsme_mean = 100 * rmse / (denormalYTest.mean())

        return mae, nmae, rmse, nrsme_maxMin, nrsme_mean

    def drawGraphStation(self, station, visualise = 1, ax = None ):
        '''draw graph of predicted vs real values'''

        yTest = self.yTest[:, station]
        denormalYTest = self.denormalize(yTest)

        denormalPredicted = self.denormalize(self.predicted[:, station])

        mae, nmae, rmse, nrmse_maxMin, nrmse_mean  = self.errorMeasures(denormalYTest, denormalPredicted)
        print('station %s : MAE = %7.7s   NMAE = %7.7s   RMSE = %7.7s   nrmse_maxMin = %7.7s   nrmse_mean = %7.7s'%(station+1, mae, nmae, rmse, nrmse_maxMin, nrmse_mean ))

        if visualise:
            if ax is None :
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(denormalYTest, label='Real')
            ax.plot(denormalPredicted, label='Predicted', color='red')
            ax.set_xticklabels([0, 100, 200, 300], rotation=40)

        return mae, nmae, rmse, nrmse_maxMin, nrmse_mean

    def drawGraphAllStations(self, cluster):
        # rows, cols = 4, 4
        maeRmse = np.zeros((self.inOutVecDim,5))

        fig, ax_array = plt.subplots(self.inOutVecDim, 1, sharex=True, sharey=True )
        staInd = 0
        for ax in np.ravel(ax_array):
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            staInd += 1
        # plt.xticks([0, 100, 200, 300])#, rotation=45)
        errMean = maeRmse.mean(axis=0)
        print(maeRmse.mean(axis=0))
        # np.savetxt("/home/longfei/Graph_LSTM/dataset/weather_1/error_results_cluster1_wea1.csv", errMean, delimiter=",", fmt='%1.12f')

        # save the predict results into csv file
        denormalReal_selected = self.denormalize(self.yTest)
        np.savetxt("/home/longfei/Graph_LSTM_All/dataset/test_results_graph_{}.csv".format(cluster), denormalReal_selected, delimiter=",", fmt='%1.12f')
        denormalPredicted_selected = self.denormalize(self.predicted)
        np.savetxt("/home/longfei/Graph_LSTM_All/dataset/predicted_results_graph_{}.csv".format(cluster), denormalPredicted_selected, delimiter=",", fmt='%1.12f')

        # filename = 'pgf/finalEpoch'
        # plt.savefig('{}.pgf'.format(filename))
        # # plt.savefig('{}.pdf'.format(filename))
        # plt.show()

        return

    def run(self):
        #  training
        xTrain, yTrain = self.loadData_1()
        print(' Training LSTM 1 ...')
        self.lstmModels[0] = self.trainLSTM(xTrain, yTrain, 1)

        for modelInd in range(1,6):
            xTrain, yTrain = self.loadData(xTrain, yTrain, self.lstmModels[modelInd-1])
            print(' Training LSTM %s ...' % (modelInd+1))
            self.lstmModels[modelInd] = self.trainLSTM(xTrain, yTrain, modelInd+1)

        # testing
        print('...... TESTING  ...')
        print(self.inOutVecDim)
        print('trainRow: %s' % (self.solars.shape[0]-11*59))
        self.test()

        self.drawGraphAllStations(self.cluster)

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
spectral_cluster = SpectralClustering(assign_labels="discretize",  affinity="rbf", random_state=0, eigen_solver='arpack').fit(weather_Data_mean)
cluster_spatio = spectral_cluster.fit_predict(weather_Data_mean) # save new clusters for chart
print("File save")
np.savetxt("/home/longfei/Graph_LSTM_All/dataset/spatio_cluster_graph.csv", cluster_spatio, delimiter=",", fmt='%1f')

for i in range (len(Counter(spectral_cluster.labels_))):
    cluster_site = [j for j, x in enumerate(cluster_spatio) if x == i]
    DeepForecaste = multiLSTM(i, cluster_site)
    DeepForecaste.run()

