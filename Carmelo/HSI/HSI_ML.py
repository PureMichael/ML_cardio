import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Input, Activation, Dropout, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, Flatten, Softmax
from keras.models import Model, Sequential

df = pd.read_csv('GT_Test.csv')
# df = pd.read_csv('MatLabData.csv')
h = np.histogram(df.iloc[:, 1], bins=np.arange(0, 8))

class Element:
    def __init__(self, data, ii):
        self.data= data.iloc[np.nonzero(data.iloc[:, 1] == ii)[0], :]
        order = random.sample(range(len(self.data)), len(self.data))
        split_idx = int(np.round(0.75*len(self.data)))
        train_set = order[:split_idx]
        test_set = order[split_idx:]
        self.train_data = self.data.iloc[train_set, :]
        self.test_data = self.data.iloc[test_set, :]


data=[]
for i in np.unique(df.iloc[:, 1]):
    data.append(Element(df, int(i)))

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for i in np.arange(len(data)):
    train_data = train_data.append(data[i].train_data)
    test_data = test_data.append(data[i].test_data)

order_train = random.sample(range(len(train_data)), len(train_data))
order_test = random.sample(range(len(test_data)), len(test_data))
train_labels_og = train_data.iloc[order_train, 1].values
test_labels_og = test_data.iloc[order_test, 1].values
train_data = train_data.iloc[order_train, 2:].values
test_data = test_data.iloc[order_test, 2:].values
train_labels = to_categorical(train_labels_og)
test_labels = to_categorical(test_labels_og)
num_inputs = train_data.shape[1]
num_outputs = test_labels.shape[1]

model = Sequential()
model.add(Dense(num_inputs, input_dim=(num_inputs), activation='relu'))
# model.add(Dense(num_inputs*2, activation='relu'))
# model.add(Dense(num_inputs*10, activation='relu'))
# model.add(Dense(num_inputs*2, activation='relu'))
model.add(Dense(num_outputs, activation='sigmoid'))
omt = keras.optimizers.Adam(lr=0.00001)
model.compile(optimizer=omt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels,
          epochs =25,
          verbose=1,
          batch_size=int(len(test_labels_og)/50))
prd = model.predict_classes(test_data)
scores = model.evaluate(test_data, test_labels)
print(str(model.metrics_names[1])+' %.2f%%' % (scores[1]*100) + ' accuracy on test data')
print(np.column_stack((prd[0:10], test_labels_og[0:10])))
plt.figure()
ax1 = plt.subplot(2,1,1)
plt.plot(test_labels_og)
ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(prd)
plt.show()








