import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
 
# data csv columns:  Flow	PressureIN	DNIraw	DNIreal	TIN	TOUT	DeltaT	JP_Fx 
DATA = pd.read_csv('./dataCS.csv', delimiter=';', na_values=['NaN'])

# predict value
FLOW_SF =  DATA['Flow'].astype(float).values.reshape(-1, 1) 

# predictor values 
# for test, change manually to "inputData" and "weight" 
PREASSURE_IN =  DATA['PressureIN'].astype(float).values 
DNI_RAW =  DATA['DNIraw'].astype(float).values 
DNI_REAL =  DATA['DNIreal'].astype(float).values 
T_IN =  DATA['TIN'].astype(float).values 
T_OUT =  DATA['TOUT'].astype(float).values 
DELTA_T =  DATA['DeltaT'].astype(float).values 
DELTA_T = np.clip(DELTA_T, a_min=0, a_max=None)
JP_FX =  DATA['JP_Fx'].astype(float).values   

##################################################################
# Change value <DNI_REAL> to TRAIN MODEL: 
inputDataTrain = DNI_REAL 
 
# Change value <T_OUT> to TRAIN MODEL: 
weightTrain = DELTA_T   
 
# Change value of "inputTest" to TEST MODEL:
inputTest = 999   
##################################################################
   

TESTDATA = inputTest # hold before reshape data

# reshape to 2d matrix 
inputDataTrain = inputDataTrain.reshape(-1, 1) 
inputTest = np.array([inputTest]).reshape(-1, 1) 
  

# instance model 
model = LinearRegression()   

# train model
model.fit(inputDataTrain, FLOW_SF, sample_weight = weightTrain) 
 
# calculate result 2D
result = model.predict(inputTest)

 
# print results
print('+++++++++++++++') 
print('MODEL OUTPUT: ', result) 
print('+++++++++++++++') 
 
# simple linear regresion for check
# y = a * x + b
# (inputTest *  modelo.coef_[0]) + modelo.intercept_ 
# weighted formula :
# y = (sum(wi)*sum(wi * xi * yi) - sum(wi * xi)*sum(wi * yi)) / (sum(wi)*sum(wi * xi^2) - (sum(wi * xi))^2) * x + (sum(wi * yi) - b * sum(wi * xi)) / sum(wi)  
print('+++++++++++++++') 
print('coef:', model.coef_[0])
print('intercept :', model.intercept_)  
print('Input value: ', TESTDATA)
print('Output value [unweighted value, just for test]: ', (TESTDATA *  model.coef_[0]) + model.intercept_)
print('+++++++++++++++') 
  

# Graphic plot
plt.scatter(inputDataTrain, FLOW_SF, color='blue')
plt.plot(inputDataTrain, model.predict(inputDataTrain), color='red', linewidth=3)
plt.title('Caudal vs Radiación')
plt.xlabel('radiación')
plt.ylabel('caudal')
plt.show() 




#Save and load model trained functions
"""
with open('modelo_entrenado.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)
"""

""""
with open('modelo_entrenado.pkl', 'wb') as archivo:
    pickle.dump(modelo, archivo)
"""