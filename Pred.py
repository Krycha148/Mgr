import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin")
os.add_dll_directory("D:/CUDnn/cudnn-windows-x86_64-8.9.6.50_cuda12-archive/bin")



features_columns = ['p (mbar)', 'VPmax (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'rho (g/m**3)', 'wv (m/s)']
target_column = "T (degC)"
all_columns = features_columns + [target_column]
split_fraction = 0.8
sampling_rate = 6
batch_size = 128
epochs = 10
sequence_length = 20
learning_rate = 0.001

data= pd.read_csv("D:/Dokumenty/SEM2M/M/jena_climate_2009_2016.csv")

train_split = int(split_fraction * int(data.shape[0]))
val_split = int((int(data.shape[0])-train_split)/2)

train_data = data.iloc[0: train_split - 1][all_columns]
val_data = data.iloc[train_split:train_split+val_split-1][all_columns]
test_data = data.iloc[train_split+val_split:][all_columns]

mean = train_data[features_columns].mean() 
std = train_data[features_columns].std()
train_data[features_columns] = (train_data[features_columns] - mean) / std
val_data[features_columns] = (val_data[features_columns] - mean) / std
test_data[features_columns] = (test_data[features_columns] - mean) / std

x_train = train_data[features_columns].values
y_train = train_data[target_column].values

x_val = val_data[features_columns].values
y_val = val_data[target_column].values

x_test = test_data[features_columns].values
y_test = test_data[target_column].values

train_ds = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=sampling_rate,
    batch_size=batch_size,
)
val_ds = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=sampling_rate,
    batch_size=batch_size,
)

pred_ds = keras.preprocessing.timeseries_dataset_from_array(
    x_test,
    None,
    sequence_length=sequence_length,
    sampling_rate=sampling_rate,
    batch_size=batch_size,
)



model = keras.Sequential([
    keras.layers.Input(shape=(sequence_length, len(features_columns))),
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dense(32, activation="swish"),
    keras.layers.Dense(1)
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mape", "mae"])



print(model.summary())


history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds
)


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid()
plt.show()

predicts = model.predict(pred_ds)
print("pred1:",len(predicts))
plt.figure(2)
plt.title("Prediction")
plt.plot(y_test[:100], label='True Values', color='green')
plt.plot(predicts[:100], label='Pred', color='blue')
plt.legend(['True Values', 'Prediction'], loc='upper right')
plt.grid()
plt.show()

# predicts2 = model.predict(pred_ds2)
# print("pred2:",len(predicts2))
# # plt.figure(3)
# # plt.title("Prediction")
# # plt.plot(y_train[1000:1400], label='True Values', color='green')
# # plt.plot(predicts2, label='Pred', color='blue')
# # plt.legend(['True Values', 'Prediction'], loc='upper right')
# # plt.grid()
# # plt.show()

# predicts3 = model.predict(pred_ds3)
# print("pred3:",len(predicts3))
# # plt.figure(2)
# # plt.title("Prediction")
# # plt.plot(y_train[3000:3400], label='True Values', color='green')
# # plt.plot(predicts3, label='Pred', color='blue')
# # plt.legend(['True Values', 'Prediction'], loc='upper right')
# # plt.grid()
# # plt.show()

