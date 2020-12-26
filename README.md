Graph Machine Learning Package for PV Generation Forecasting:

Step 0. Prepare PV generation data and weather data as formats:
•	PV generation data: (./data/solar_data.csv)
•	Weather data: (./data/W_train.csv)

Step 1. Outlier Detection:
•	Code: python3 Outlier_Com.py
•	Function: Unsupervised Outlier Detection using OneclassSVM and Isolation Forest algorithms for spatio-termal series data
•	Result: (./data/solar_data_preprocessed.csv)

Step 2. Interpolation:
•	Code: python3 Interp.py
•	Function: Linear Interpolation for spatio-termal series data
•	Result: (./data/solar_data_final.csv)

Step 3. Spectral Clustering:
•	Code: python3 Spatio_Cluster.py
•	Function: kmeans clustering for spatio-termal series data based on the weather data
•	Result: (. /data/ spatio_cluster.csv)

Step 4. Graph 5-layer LSTM model training:
•	Code: python3 GraphLSTM.py
•	Function: train the multiLSTM model for spatio-termal series data

