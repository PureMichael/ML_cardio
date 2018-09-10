import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import random

random.seed(6)

xl = pd.ExcelFile('CTG.xls')
mldata = xl.parse('MLData')
print(mldata.shape)
split = round(0.90*int(mldata.shape[0]))
order = random.sample(range(mldata.shape[0]), mldata.shape[0])
data = np.empty(shape=(int(mldata.shape[0]), int(mldata.shape[1])))

for ii in order:
    data[ii, :] = mldata.iloc[order[ii], 0:]

trainData = data[0:split, :-1]
trainLabels = data[0:split, -1]
testData = data[split:int(data.shape[0]), 0:-1]
testLabels = data[split:int(data.shape[0]), -1]

# Create the Model
model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model
model.add(keras.layers.Dense(10, input_dim=(data.shape[1]-1), activation='relu'))
# Anotha one
model.add(keras.layers.Dense(10, activation='sigmoid'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Set up training
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mse',
              metrics=['accuracy'])

model.fit(trainData, trainLabels, epochs=50, batch_size=32,
          validation_data=(testData, testLabels))

scores = model.evaluate(trainData, trainLabels)
print("\n%sL %.2f%%" % (model.metrics_names[1], scores[0]*100))





