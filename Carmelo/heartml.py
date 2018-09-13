import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import datetime
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
# random.seed(6)
# tf.set_random_seed(333)
# np.random.seed(856)
# xl = pd.ExcelFile('AllScaled.xls')
xl = pd.ExcelFile('CTG.xls')
# xl = pd.ExcelFile('EvenDistribution_Scaled.xls')
mldata = xl.parse('MLData')
print(mldata.shape)
split = round(0.75*int(mldata.shape[0]))
trainedDataAccuracy = np.zeros(10)
trainedDataAccuracy[:] = np.nan
testDataAccuracy = np.zeros(10)
testDataAccuracy[:] = np.nan
runTime=[]
a = datetime.now()
order = random.sample(range(mldata.shape[0]), mldata.shape[0])
data = np.empty(shape=(int(mldata.shape[0]), int(mldata.shape[1])))
for ii in order:
    data[ii, :] = mldata.iloc[order[ii], 0:]
data[:,-1]=np.subtract(data[:,-1],1)
trainData = data[0:split, :-2]
trainLabels = data[0:split, -1]
m, n = data.shape
for i in range(m):
    if data[i,-1] == 1:
        color='r.'
    elif data[i,-1] == 2:
        color='g.'
    else:
        color='b.'
    plt.plot(np.arange(n),data[i,:], color)
plt.show()
print(np.column_stack((trainData, trainLabels)))
testData = data[split:int(data.shape[0]), 0:-2]
testLabels = data[split:int(data.shape[0]), -1]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(trainLabels)
encoded_Y = encoder.transform(trainLabels)
# convert integers to dummy variables (i.e. one hot encoded)
hot_y = keras.utils.to_categorical(encoded_Y)
print(hot_y)
print(hot_y.shape)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(testLabels)
encoded_Y2 = encoder.transform(testLabels)
# convert integers to dummy variables (i.e. one hot encoded)
hot_y2 = keras.utils.to_categorical(encoded_Y2)
# Create the Model
model = keras.Sequential()
model.add(keras.layers.Dense(trainData.shape[1], input_dim=(trainData.shape[1]), activation='relu'))
model.add(keras.layers.Dense(trainData.shape[1]*2, activation='relu'))
model.add(keras.layers.Dense(trainData.shape[1]*3, activation='relu'))
model.add(keras.layers.Dense(trainData.shape[1]*4, activation='relu'))
model.add(keras.layers.Dense(trainData.shape[1]*3, activation='relu'))
model.add(keras.layers.Dense(trainData.shape[1]*2, activation='relu'))
model.add(keras.layers.Dense(trainData.shape[1], activation='relu'))
model.add(keras.layers.Dense(3, activation='sigmoid'))
omt = keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer=omt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(trainData, hot_y, epochs=100, verbose=1, batch_size=20)
scores = model.evaluate(trainData, hot_y)
prd = model.predict_classes(testData)
print(np.column_stack((prd, testLabels)))
scoresTest = model.evaluate(testData, hot_y2)
print(str(model.metrics_names[1])+' %.2f%%' % (scores[1]*100) + ' accuracy on trained data')
print(str(model.metrics_names[1])+' %.2f%%' % (scoresTest[1]*100) + ' accuracy on test data')

