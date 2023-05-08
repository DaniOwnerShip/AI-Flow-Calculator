import numpy as np 
from sklearn.linear_model import LinearRegression 
import pandas as pd 
from sklearn.model_selection import train_test_split
import GraphXY
 
# data csv columns:  Flow	PressureIN	DNIraw	DNIreal	TIN	TOUT	DeltaT	JP_Fx 
DATA = pd.read_csv('./dataCS.csv', delimiter=';', na_values=['NaN'])

# use for test weight variable model
PREASSURE_IN =  DATA['PressureIN'].astype(float).values 
DNI_RAW =  DATA['DNIraw'].astype(float).values 
DNI_RAW = np.clip(DNI_RAW, a_min=0, a_max=None)  
DNI_REAL =  DATA['DNIreal'].astype(float).values 
DNI_REAL = np.clip(DNI_REAL, a_min=0, a_max=None)  
T_IN =  DATA['TIN'].astype(float).values 
T_OUT =  DATA['TOUT'].astype(float).values 
DELTA_T =  DATA['DeltaT'].astype(float).values
DELTA_T = np.clip(DELTA_T, a_min=0, a_max=None)
JP_FX =  DATA['JP_Fx'].astype(float).values   
   
# predict value
FLOW_SF =  DATA['Flow'].astype(float).values.reshape(-1, 1)  
 
# data train
DATA_TRAIN = [DNI_RAW, T_OUT, JP_FX] 
X = np.array(DATA_TRAIN).astype(float)
X = X.T 
 
#weight variable
# no effect = None
weightTrain = JP_FX # <=== CHANGE THIS VALUE  

# split data
X_train, X_test, y_train, y_test = train_test_split(X, FLOW_SF, test_size=0.2, random_state=0)  #Change random after adjust
weightTrain = weightTrain[:len(X_train)]   


# instance model
model = LinearRegression()

# train model
model.fit(X_train, y_train, sample_weight=weightTrain)

# evaluate model
score = model.score(X_test, y_test)
print("score:", score) 
 

index = 1  # MAX 145  
# test sample   
sample = X_test[index]
sample = sample.reshape(1,-1) #(n,)->(1, n)y_test

ref = y_test[index] 

# predict value
result = model.predict(sample)

diff =  result - ref  

print("result:", result)
print("ref:", ref) 
print("diff:", diff) 


# print results
print('---------------') 
print("OUTPUT FLOW =>> {:.2f}".format(result.flatten()[0]))  
print('---------------')  
print('coef :', model.coef_)
print('intercept :', model.intercept_)   
 

# invoke plot
GraphXY.plot_graph(DATA, X, model)


