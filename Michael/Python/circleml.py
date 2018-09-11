from tensorflow import keras

import math
import numpy as np
import matplotlib.pyplot as plt


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


numm=(2*math.pi)/200

t = np.empty(round(2*math.pi/numm)+1)
j = 0

for i in frange(0,2*math.pi,numm):
    t[j]=i
    j+=1


rlen=len(t)
xr1 = (np.random.rand(rlen)-0.5)/5
yr1 = (np.random.rand(rlen)-0.5)/5
x1 = np.add((0.25*np.cos(t)), xr1)
y1 = np.add(0.25*np.sin(t), yr1)
labels1 = np.zeros(len(t))
a=(0.25*np.cos(t))
# print(xr1.shape,'shape of xr1')
# print(t.shape,'shape of t')
# print(labels1.shape,'shape of labels1')
# print(x1.shape,'shape of x1')

xr2 = (np.random.rand(rlen)-0.5)/5
yr2 = (np.random.rand(rlen)-0.5)/5
x2 = np.add((np.cos(t)), xr2)
y2 = np.add(np.sin(t), yr2)
labels2 = np.zeros(len(t))

labels = np.concatenate((labels1, labels2))
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
xy=np.column_stack((x, y))
# labs=np.column_stack((labels,labels))
print(y,'shape of y')
print(xy.shape, 'shape of xy')
print(labels.shape, 'shape of lables')

plt.plot(x,y,'k.')
plt.show()

model = keras.Sequential()
model.add(keras.layers.Dense(4, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(4, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(xy,labels,epochs=50)
