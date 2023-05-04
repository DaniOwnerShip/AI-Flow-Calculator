import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler

# data csv columns:  Flow	PressureIN	DNIraw	DNIreal	TIN	TOUT	DeltaT	JP_Fx 
DATA = pd.read_csv('./dataCS.csv', delimiter=';', na_values=['NaN'])

# predict value
FLOW_SF =  DATA['Flow'].astype(float).values.reshape(-1, 1) 

# predictor values 
# for test, change manually to "inputData" and "weight" 
PREASSURE_IN =  DATA['PressureIN'].astype(float).values 
DNI_RAW =  DATA['DNIraw'].astype(float).values 
DNI_RAW = np.clip(DNI_RAW, a_min=0, a_max=None)
DNI_REAL =  DATA['DNIreal'].astype(float).values 
T_IN =  DATA['TIN'].astype(float).values 
T_OUT =  DATA['TOUT'].astype(float).values 
DELTA_T =  DATA['DeltaT'].astype(float).values
DELTA_T = np.clip(DELTA_T, a_min=0, a_max=None)
JP_FX =  DATA['JP_Fx'].astype(float).values   

##################################################################
# Change values to TRAIN MODEL:  
 
inputDataTrain = DATA[['DNIreal', 'TOUT', 'JP_Fx', 'DeltaT', 'PressureIN']]  

 
# Change value <T_OUT> to TRAIN MODEL: 
weightTrain =  JP_FX 

# Change values to TEST MODEL:
AtestData = ['797.36', '387.44', '7.07', '93.4', '28.27'] 
AtestData = np.array([AtestData])  # Convertir la lista a una matriz de numpy con una sola fila
AtestData = AtestData.astype(float)
 
##################################################################
    
# instance model 
model = LinearRegression()   

# train model
model.fit(inputDataTrain, FLOW_SF, sample_weight = weightTrain) 
 
# calculate result 2D
result = model.predict(AtestData)

 
# print results
print('***************') 
print('MODEL OUTPUT: ', result)  
print('coef :', model.coef_)
print('intercept :', model.intercept_)   
print('***************') 
  
"""
# Graphic plot
plt.scatter(inputDataTrain, FLOW_SF, color='blue')
plt.plot(inputDataTrain, model.predict(inputDataTrain), color='red', linewidth=3)
plt.title('Caudal vs Radiación')
plt.xlabel('radiación')
plt.ylabel('caudal')
plt.show() 




#Save and load model trained functions

with open('modelo_entrenado.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)
"""

""""
with open('modelo_entrenado.pkl', 'wb') as archivo:
    pickle.dump(modelo, archivo)
"""