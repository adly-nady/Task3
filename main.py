import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras as kr


df = pd.read_csv('kc_house_dataset.csv')

df = df.drop('id',axis=1)

df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].apply(lambda date:date.month)

df['year'] = df['date'].apply(lambda date:date.year)

df = df.drop('date',axis=1)

df = df.drop('zipcode',axis=1)

x = df.drop('price',axis=1)

y = df['price']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

datasets = x_test.copy()
datasets['real_price'] = y_test;


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
 

def print_evaluate(true,predicted,train=True):
    mae = metrics.mean_absolute_error(true, predicted)

    mse = metrics.mean_squared_error(true, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))

    r2_square = metrics.r2_score(true, predicted)

    
    if train:

        print("================| training |=======================")
        print(f" MAE:{mae} \n MSE:{mse} \n RMSE:{rmse} \n accuracy:{r2_square}")

    else:

        print("================| testing |=======================")
        print(f" MAE:{mae} \n MSE:{mse} \n RMSE:{rmse} \n accuracy:{r2_square}")        


model = kr.Sequential()

model.add(kr.layers.Dense(x_train.shape[1]))

model.add(kr.layers.Dense(128,activation='relu')) 
model.add(kr.layers.Dense(128,activation='relu')) 
model.add(kr.layers.Dense(128,activation='relu'))
model.add(kr.layers.Dense(128,activation='relu'))
model.add(kr.layers.Dense(128,activation='relu')) 
model.add(kr.layers.Dense(128,activation='relu')) 
model.add(kr.layers.Dense(128,activation='relu'))
model.add(kr.layers.Dense(64,activation='relu'))
model.add(kr.layers.Dense(4,activation='relu'))

model.add(kr.layers.Dropout(0.0115))
model.add(kr.layers.Dense(1))


model.compile(optimizer='adam',loss='mse')
results = model.fit(x_train,y_train.values,validation_data=(x_test,y_test.values),epochs=129)

plt.figure(figsize=(10,6))

plt.plot(results.history['loss'],label='loss')
plt.plot(results.history['loss'],label='val_loss')

# plt.legend()
plt.show()

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

datasets['predict_price'] = y_test_pred;

df = pd.DataFrame(datasets)
df.to_csv('output_model.csv', index = False, encoding='utf-8') 

print_evaluate(y_train,y_train_pred, train=True)
print_evaluate(y_test,y_test_pred, train=False)

