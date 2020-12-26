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

class drclass(object):
    def _collap(self,vec):
        return "".join(str(x) for x in vec)
        
    def _outersum1(self,indvec,y,lam):
        ytemp = y[:,indvec]
        resmat = np.matmul(ytemp,ytemp.transpose())
        resmat = resmat + np.identity(y.shape[0])*lam
        return resmat
        
    def _linter(self,vec):
        x = np.linspace(0,len(vec)-1,len(vec))
        good = np.where(np.isfinite(vec))
        if len(good[0])>1:
            f = interp1d(x[good],vec[good],bounds_error=False)
            return f(x)
        else:
            return vec
        
    def linterp(self,mat):
        if (mat.shape[0]==0 or mat.shape[1]==0):
            return mat
        dim=mat.shape[1]
        mat1 = np.apply_along_axis(self._linter,1,mat)
        mat1 = np.concatenate((mat1[:,int(dim/2):dim],mat1[:,0:int(dim/2)]),axis = 1)
        mat1 = np.apply_along_axis(self._linter,1,mat1)
        mat1 = np.concatenate((mat1[:,int(dim/2):dim],mat1[:,0:int(dim/2)]),axis = 1)
        return mat1
        
    def modifiedals(self, mat, rank, mat1=None, lam = 3, mu = 1, thres = 0.001, maxit = 20):
        y = np.random.randn(rank,mat.shape[1])
        x = np.empty((rank,mat.shape[0]))
        indmat = ~np.isnan(mat)
        
        j = 1
        dist = 1000
        rec = 0
        
        while(j<maxit and dist>thres):
            for i in range(mat.shape[0]):
                x[:,i] = np.matmul(np.linalg.inv(np.matmul(y[:,indmat[i]],y[:,indmat[i]].transpose())+lam*np.identity(rank)),np.matmul(y[:,indmat[i]],mat[i,indmat[i]]))
            y[:,0] = np.matmul(np.linalg.inv(self._outersum1(indmat[:,0],x,lam+mu)),(np.matmul(x[:,indmat[:,0]],mat[indmat[:,0],0])+mu*y[:,1]))
            for i in range(1,y.shape[1]-1):
                y[:,i] = np.matmul(np.linalg.inv(self._outersum1(indmat[:,i],x,lam+2*mu)),(np.matmul(x[:,indmat[:,i]],mat[indmat[:,i],i])+mu*(y[:,i-1]+y[:,i+1])))
            y[:,y.shape[1]-1] = np.matmul(np.linalg.inv(self._outersum1(indmat[:,y.shape[1]-1],x,lam+mu)),(np.matmul(x[:,indmat[:,y.shape[1]-1]],mat[indmat[:,y.shape[1]-1],y.shape[1]-1])+mu*y[:,y.shape[1]-2]))
            
            test = np.matmul(x.transpose(),y)
            dist = abs(rec- np.nansum(abs(test-mat))/sum(sum(indmat)))
            rec = np.nansum(abs(test-mat))/sum(sum(indmat))
            print('Training Error: ..... ', rec)
            if mat1:
                testerror = np.nanmean(abs(test - mat1)[~indmat])
                print('Testing Error: ...... ',testerror)
            print('dpimprove_all Finished Iteration No. ',j)
            j = j+1
        return x, y

def run_algo(runtype='sas',rank=15,lam=1.5,mu=2,numclus=200,take=5,num=40):
        print('run_algo')
        print(runtype)
        algostr=""
        conn=None
        try:
            inOutVecDim = 273  # number of stations
            file_dataset = '/home/longfei/Graph_LSTM_All/dataset/solar_data_preprocessed.csv'
            with open(file_dataset) as f:
                data = csv.reader(f, delimiter=",")
                solar = []
                for line in data:
                    solar.append((line))
            solar = (np.array(solar)).astype(float) # all data
            solar = solar[:,:inOutVecDim]
            solarc = solar.copy()
            solarc[np.nanmean(solarc,axis=1)==0]=0
            solarc[solarc<0]=0
            solarc1 = solarc[np.nanmean(solarc,axis=1)!=0]
            if runtype=='sas':
                algostr="Smoothed Alternating Least Squares"
                x, y = drclass().modifiedals(mat = solarc1, rank = rank, mat1=None, lam = lam, mu = mu, thres = 0.00001, maxit=20)
                res = np.matmul(x.transpose(),y)

            # elif runtype=='karm':
            #     datt = pd.DataFrame({'meterid':dat[0].astype(str)+"_"+dat[0].astype(str),
            #                          'mdate':dat[1]})
            #     datt = datt[['meterid','mdate']]
            #     res = drclass().karm(datt=datt,matt=matt1,matt1=None,num=num)
            # elif runtype=='cbms':
            #     res = drclass().corplug(mat1=matt1, take=take, numclus= numclus,read=1)
            elif runtype == 'li':
                algostr="Linear Interpolation"
                res = drclass().linterp(matt1)  
            # elif runtype=='karm':
            #     datt = pd.DataFrame({'meterid':dat[0].astype(str)+"_"+dat[0].astype(str),
            #                          'mdate':dat[1]})
            #     datt = datt[['meterid','mdate']]
            #     res = drclass().karm(datt=datt,matt=matt1,matt1=None,num=num)
            # elif runtype=='cbms':
            #     res = drclass().corplug(mat1=matt1, take=take, numclus= numclus,read=1)
            # else:
            #     algostr="Linear Interpolation"
            #     # f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\t运行"+algostr+"算法\n")
            #     # f.flush()
            #     res = drclass().linterp(matt1)        
            
            # f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\t"+algostr+"算法运行完毕，写入数据库\n")
            # f.flush()
            solarc[np.nanmean(solarc,axis=1)!=0]=res
            solarc[solarc<0]=0
            solar = solarc
            print('...... TESTING  ...')
            np.savetxt("/home/longfei/Graph_LSTM_All/dataset/solar_data_final.csv", solar, delimiter=",", fmt='%1.12f')
        except (Exception) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return 0
run_algo()