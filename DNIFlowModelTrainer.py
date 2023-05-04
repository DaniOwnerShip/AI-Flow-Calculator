import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
 
# data csv columns:  Flow	PressureIN	DNIraw	DNIreal	TIN	TOUT	DeltaT	JP_Fx 
DATA = pd.read_csv('./dataCS.csv', delimiter=';', na_values=['NaN'])

# predict value
FLOW_SF =  DATA['Flow'].astype(float).values.reshape(-1, 1) 

# predictor values  
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

#Change name of rows to test model
inputDataTrain = DATA[['DNIreal', 'TOUT', 'JP_Fx', 'DeltaT', 'PressureIN']]    

#select the column of Data Test file to test. must be the same order like Data Train
#in this case Test data are data of Train data:

rowTest = 356 # <=== CHANGE THIS VALUE  

rowTest = rowTest + 2 #add 2 row (header + start 1) to match row of ./dataCS.csv

# print data
print(inputDataTrain.shape)
print(inputDataTrain.iloc[rowTest].values)
  
dataTest = inputDataTrain.iloc[rowTest].to_numpy().astype(float).reshape(1, -1)

#weight variable
weightTrain =  JP_FX # no effect = None
 
##################################################################
    

# instance model 
model = LinearRegression()   

# train modelmodel = LinearRegression() 
model.fit(inputDataTrain.values, FLOW_SF, sample_weight=weightTrain)


# calculate result   
result = model.predict(dataTest)

 
# print results
print('***************') 
print('MODEL OUTPUT: ', result)  
print('---------------') 
print('coef :', model.coef_)
print('intercept :', model.intercept_)   
print('***************') 




 