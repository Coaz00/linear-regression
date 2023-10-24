import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Ucitavanje i prikaz podataka
data = pd.read_csv("fuel_consumption.csv")

pd.set_option("display.max_columns", 13)
pd.set_option("display.width", 50)

print(data.head(5))
print(data.info())
print(data.describe())

#Popunjavanje i eliminacija nedostajucih podataka
data['ENGINESIZE'] = data['ENGINESIZE'].fillna(np.mean(data['ENGINESIZE']))
data.dropna(inplace=True) # TRANSMISSION(3) and FUELTYPE(2)
data.reset_index(drop=True,inplace = True)

# ohe encoding kategorickih atributa
def ohe_encoding(feature):
    ohe = OneHotEncoder(dtype = int, sparse = False)
    fuel_type = ohe.fit_transform(data[feature].to_numpy().reshape(-1,1))
    data.drop(columns = [feature],inplace = True)
    return data.join(pd.DataFrame(data = fuel_type, columns = ohe.get_feature_names([feature])))





print(data.info())

# kontinualni atributi
data_cont = ['ENGINESIZE','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']

# Korelaciona matrica
plt.figure()
sb.heatmap(data[data_cont].corr(), annot= True)
plt.show()

output = data['CO2EMISSIONS']
data = data.drop(columns = ['CO2EMISSIONS'])

# Zavisonst izlaza od kontinualnih atributa
def decart2D(feature):
    plt.figure()
    plt.scatter(data[feature],output)
    plt.ylabel('CO2 emission')
    if(feature == 'ENGINESIZE'):
        plt.xlabel(feature + '[m^3]')
    elif(feature == 'FUELCONSUMPTION_COMB_MPG'):
        plt.xlabel(feature + '[mile/gallon]')
    else:
        plt.xlabel(feature + '[l/100km]')
    plt.show()

for feature in data_cont[:-1]:
    decart2D(feature)
    



    
liters_in_gallon = 3.78541178
miles_in_km = 1.609344

# miles/gallon -> liters/100km
data['FUELCONSUMPTION_COMB_MPG'] = 1/data['FUELCONSUMPTION_COMB_MPG']*liters_in_gallon/miles_in_km*100
plt.figure()
plt.scatter(data['FUELCONSUMPTION_COMB_MPG'],output)
plt.ylabel('CO2 emission')
plt.xlabel('FUELCONSUMPTION_COMB_MPG[l/100km]')
plt.show()

# izbacivanje neinformativnih atributa
data = data.drop(columns = ['MODELYEAR','MODEL','MAKE','TRANSMISSION','FUELCONSUMPTION_COMB','VEHICLECLASS','FUELTYPE'])

#data = ohe_encoding('FUELTYPE')
#data = ohe_encoding('MAKE')
# data = ohe_encoding('MODEL')
#data = ohe_encoding('VEHICLECLASS')
#data = ohe_encoding('TRANSMISSION')

# scaler = MinMaxScaler()
# scaler.fit(data)
# data = pd.DataFrame(scaler.transform(data),columns = data.columns)

# Podela na trening/test skup
X_train,X_test,Y_train,Y_test = train_test_split(data,output,test_size = 0.2,shuffle = True, random_state = 53)

# Built-in LR
lr = LinearRegression()
lr.fit(X_train,Y_train)
Y_pred_test = lr.predict(X_test)
Y_pred_train = lr.predict(X_train)

mse_test = 1/2/len(Y_pred_test)*np.sum((Y_pred_test - Y_test)**2)
#mse_train = 1/2/len(Y_pred_train)*np.sum((Y_pred_train - Y_train)**2)
print("------------")
print("Built-in LR")
print("------------")

print("Tezine: ")
print(lr.coef_.reshape(-1,1))
print("Cost na test skupu: " + str(mse_test))




# Moja Linearna Regresija

# predikcija izlaza na osnovu tezina i atributa
def predict(weights,X):
    pred = weights[0] + X@weights[1:]
    return pred[0]

# racunanje cost-a na test skupu
def cost(weights,X_test,Y_test):
    s = 0
    for i in range(len(X_test)):
        s += (predict(weights,X_test.iloc[i,:]) - Y_test.iloc[i])**2
    
    return 1/2/len(X_test)*s

# racunanje update-a za tezine
def calc_update(weights,X_train,Y_train):
    update = np.zeros(len(weights)).reshape(-1,1)
    preds = np.zeros(len(X_train)).reshape(-1,1)
    # racunanje predikcija za trenutne tezine
    for i in range(len(X_train)):
        preds[i] = predict(weights,X_train.iloc[i,:])
    # racunanje update-a za w0
    for i in range(len(preds)):
        update[0] += preds[i] - Y_train.iloc[i]
    update[0] = update[0]/len(preds)
    
    # racunanje update-a za  w1 -> wn
    for k in range(len(X_train.columns)):
        for i in range(len(preds)):
            update[k+1] += (preds[i] - Y_train.iloc[i])*X_train.iloc[i,k]
        update[k+1] = update[k+1]/len(preds)
    
    return update

# treniranje modela
def train(X_train,Y_train,X_test,Y_test,learning_rate,max_epochs,max_cost):
    # inicijalizacija tezina
    weights = np.zeros(len(data.columns)+1).reshape(-1,1)
    
    # update tezina
    for i in range(max_epochs):
        weights = weights - learning_rate*calc_update(weights,X_train,Y_train)
        # uslov za izlazak ako dostignemo zeljenu gresku
        # if(cost(weights,X_test,Y_test) < max_cost):
        #     return weights
    
    return weights
# Training
learning_rate = 3e-3
max_epochs = 20
max_cost = 10
weights = train(X_train,Y_train,X_test,Y_test,learning_rate,max_epochs,max_cost)

# Testing
print("------------")
print("Moj LR")
print("------------")
print("Tezine: ")
print(weights)
print("Cost na test skupu: " + str(cost(weights,X_test,Y_test)))

