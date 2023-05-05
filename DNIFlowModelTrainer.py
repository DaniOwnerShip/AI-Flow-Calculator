import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime 
 
# data csv columns:  Flow	PressureIN	DNIraw	DNIreal	TIN	TOUT	DeltaT	JP_Fx 
DATA = pd.read_csv('./dataCS.csv', delimiter=';', na_values=['NaN'])

# predict value
FLOW_SF =  DATA['Flow'].astype(float).values.reshape(-1, 1) 

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
  
##################################################################  
#INDEX csv: 'Time' | 'Flow' | 'PressureIN' | 'DNIraw' | 'DNIreal' | 'TIN' | 'TOUT' | 'DeltaT' | 'JP_Fx' 
#Change name of rows to test model 
inputDataTrain = DATA[['DNIreal', 'TOUT',  'JP_Fx']]    # <=== CHANGE THIS VALUE. colums number can be change too

#weight variable
# no effect = None
weightTrain = JP_FX # <=== CHANGE THIS VALUE  

#select the column of Data Test file to test. must be the same order like Data Train
#in this case Test data are from Train data file:
#row of ./dataCS.csv
rowTest = 385 # <=== CHANGE THIS VALUE    
rowTest = rowTest + 2 # just to match row of ./dataCS.csv (header + start 1)  

# print data 
print('***************') 
print(inputDataTrain.shape) 
print('Image all test data: ', DATA.iloc[rowTest].keys)
print('Used test data: ', inputDataTrain.iloc[rowTest].values)
flowTraining = DATA['Flow'][rowTest]
print('Flow training => ', flowTraining)
print('DNIe  training=> ', DATA['DNIreal'][rowTest]) 
 
################################################################## 

# instance model 
model = LinearRegression()   

# train model 
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
 

# xyplot  
dni = np.array(DATA['DNIreal']).reshape(-1, 1) 
Tout = np.array(DATA['TOUT']).reshape(-1, 1) 
Tin = np.array(DATA['TIN']).reshape(-1, 1)  

fig, ax1 = plt.subplots(figsize=(16, 6))

ax1.plot(DATA['Time'], FLOW_SF, color='blue', linewidth=3)
ax1.plot(DATA['Time'], model.predict(inputDataTrain.values), color='red', linewidth=3) 

ax2 = ax1.twinx()
ax2.plot(DATA['Time'], dni, color='yellow', linewidth=2)
ax2.set_ylabel('DNI') 

ax3 = ax1.twinx()
ax3.plot(DATA['Time'], Tout, color='black', linewidth=1) 
ax3.spines['right'].set_position(('axes', 1.1))
ax3.set_ylabel('Tout') 
ax3.yaxis.set_label_coords(1.08, 0.5)   
ax3.set_ylim(100, 400)  

ax4 = ax1.twinx()
ax4.plot(DATA['Time'], Tin, color='gray', linewidth=1) 
ax4.spines['right'].set_position(('axes', 1.1))
ax4.set_ylabel('Tin') 
ax4.yaxis.set_label_coords(1.1, 0.5)   
ax4.set_ylim(100, 400)

ax1.set_title('Caudal real vs Modelo')
ax1.set_xlabel('tiempo')
ax1.set_ylabel('caudal')
ax1.set_xticks(DATA['Time'][::30])
 

# Agregar cursor
time_float = []
for t in DATA['Time']:
    time_obj = datetime.strptime(t, '%H:%M')
    time_float.append(time_obj.hour * 3600 + time_obj.minute * 60)
    
cursor = ax1.axvline(x=0, color='black', linewidth=1, linestyle='--')
textpos = ax1.text(0.01, 0.7, '', transform=ax1.transAxes, ha='left')


def validateX(x):
    x = x if x is not None else None
    x = max(0, min(x, len(time_float)-1)) if x is not None else None
    return x


def on_move(event):
    x, y = event.xdata, event.ydata
    x = validateX(x)
    
    if validateX(x) == None :
        return 

    cursor.set_xdata([x]) 
    
    index = round(x)     
    y1 = FLOW_SF[index]
    y2 = model.predict(inputDataTrain.values)[index]
    y3 = dni[index]
    y4 = Tin[index]  
    y5 = Tout[index]  

    qdif = y1 - y2
    dt = y5 - y4
    
    textpos.set_text('Hora: ' + DATA['Time'][index] + 
                     '\nCaudal real: ' + str(y1.round(2)) + 
                     '\nCaudal modelo: ' + str(y2.round(2)) + 
                     '\nCaudal dif(r-m): ' + str(qdif.round(2)) + 
                     '\nDNI: ' + str(y3.round(2)) + 
                     '\nTin: ' + str(y4.round(2))+ 
                     '\nTout: ' + str(y5.round(2))+ 
                     '\nDT: ' + str(dt.round(2)) )
       

    fig.canvas.draw_idle()


fig.canvas.mpl_connect('motion_notify_event', on_move) 

plt.show()

 

