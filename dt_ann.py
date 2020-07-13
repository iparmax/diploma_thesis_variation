import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dropout

# Import data from csv
events = pd.read_csv("master_table.csv")

# Determining dependent Variable
y = events.pop("Harsh_Events_Rate")

# Determining Independent Variables
x = events.iloc[:,1:-8]

# Feature Scaling
sc = StandardScaler()
x = sc.fit_transform(x)

# Splitting dataset to train_set, test_set and validation_set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 0)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.3,random_state = 0)

# Building the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=12,activation="relu"))
ann.add(Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=12,activation="relu"))
ann.add(Dropout(0.2))
ann.add(tf.keras.layers.Dense(1))

# Compiling the ANN
opt = keras.optimizers.Adam(learning_rate=0.005)
ann.compile(optimizer=opt,loss='mean_squared_error')

# Training the ANN
r = ann.fit(x_train,y_train, batch_size=1, validation_data =(x_val,y_val),epochs = 200)

# Plotting validation loss/loss per epoch
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

# Making predictions for the test values
predicted = ann.predict(x_test)
real_values = y_test.values

# Plotting Predictions vs Test Values
junctions = y_test.index.values + 1
labels = [f'J{x}' for x in junctions]
fig, ax = plt.subplots()
fig.canvas.draw()
plt.xticks(np.arange(1, junctions.shape[0], 1,))
plt.xlim(1, junctions.shape[0])
plt.plot(real_values, color = 'red', label = 'Real ')
plt.plot(predicted, color = 'blue', label = 'Predicted')
ax.set_xticklabels(labels)
plt.title('Prediction vs Real Values')
plt.xlabel('Junctions')
plt.ylabel('Mean Harsh Event Rate')
plt.legend()
plt.show()

# Presenting Accuracy Measures
r2 = r2_score(real_values, predicted)
print(f'R-squared of ANN is {r2}')

rmse = mean_squared_error(real_values, predicted)
print(f'Root Mean Squared Error of ANN is {rmse**0.5}')

