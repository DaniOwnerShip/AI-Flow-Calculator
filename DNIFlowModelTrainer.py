import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
 
# data csv columns:  Flow	PressureIN	DNIraw	DNIreal	TIN	TOUT	DeltaT	JP_Fx 
DATA = pd.read_csv('./dataCS.csv', delimiter=';', na_values=['NaN'])

# predict value
FLOW_SF =  DATA['Flow'].astype(float).values.reshape(-1, 1) 

# use for test weight variable model
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
#INDEX csv: Time | Flow | PressureIN | DNIraw | DNIreal | TIN | TOUT | DeltaT | JP_Fx

#Change name of rows to test model 
inputDataTrain = DATA[['DNIreal', 'TOUT', 'TIN', 'DeltaT', 'DNIraw', 'PressureIN', 'JP_Fx']]     # <=== CHANGE THIS VALUE. colums number can be change too

#select the column of Data Test file to test. must be the same order like Data Train
#in this case Test data are data of Train data:

#row of ./dataCS.csv
rowTest = 385 # <=== CHANGE THIS VALUE   

rowTest = rowTest + 2 # just to match row of ./dataCS.csv (header + start 1)  

# print data 
print('***************') 
print(inputDataTrain.shape)
print(inputDataTrain.iloc[rowTest].values)
flowTraining = DATA['Flow'][rowTest]
print('Flow training => ', flowTraining)
print('DNIe  training=> ', DATA['DNIreal'][rowTest])
 
#weight variable
# no effect = None
weightTrain =  JP_FX  # <=== CHANGE THIS VALUE  
 
##################################################################
    

# instance model 
model = LinearRegression()   

# train modelmodel = LinearRegression() 
model.fit(inputDataTrain.values, FLOW_SF, sample_weight=weightTrain)


# reshape dataTest to 2D matrix
dataTest = inputDataTrain.iloc[rowTest].to_numpy().astype(float).reshape(1, -1) 

# calculate result. is using data train..
result = model.predict(dataTest)

difTestTranin =  result - flowTraining  


# print results
print('---------------') 
print("OUTPUT FLOW =>> {:.2f}".format(result.flatten()[0])) 
print('---------------') 
print("Test VS Train => {:.2f}".format(difTestTranin.flatten()[0])) 
print('---------------') 
print('coef :', model.coef_)
print('intercept :', model.intercept_)   
print('***************') 



# Graphic plot  
plt.figure(figsize=(18, 6))
plt.plot(DATA['Time'], FLOW_SF, color='blue', linewidth=3)
plt.plot(DATA['Time'], model.predict(inputDataTrain), color='red', linewidth=3) 
plt.title('Caudal real vs Modelo')
plt.xlabel('tiempo')
plt.ylabel('caudal')
plt.show()
 