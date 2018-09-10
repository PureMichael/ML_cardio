import tensorflow as tf
from tensorflow import keras

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
path=os.path.dirname(os.path.realpath('circleml.py'))
print(path)

tf.set_random_seed(333)
np.random.seed(856)


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


numm=(2*math.pi)/300

t = np.empty(round(2*math.pi/numm)+1)
j = 0

for i in frange(0,2*math.pi,numm):
    t[j]=i
    j+=1


rlen=len(t)
xr1 = (np.random.rand(rlen)-0.5)/2
yr1 = (np.random.rand(rlen)-0.5)/2
x1 = np.add((0.25*np.cos(t)), xr1)
y1 = np.add(0.25*np.sin(t), yr1)
labels1 = np.zeros(len(t))
xr2 = (np.random.rand(rlen)-0.5)/2
yr2 = (np.random.rand(rlen)-0.5)/2
x2 = np.add((np.cos(t)), xr2)
y2 = np.add(np.sin(t), yr2)
labels2 = np.ones(len(t))
labels = np.concatenate((labels1, labels2))
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
xy=np.column_stack((x, y))

# plt.plot(x1,y1,'r.',x2,y2,'b.')
# plt.show()

model = keras.Sequential()
model.add(keras.layers.Dense(4, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(4, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
cblist=[keras.callbacks.ModelCheckpoint('weights.{epoch:02d}.hdf5', save_weights_only=True, period=1)]
omt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=omt,
              metrics=['accuracy'])
model.fit(xy, labels, epochs=30, verbose=1, batch_size=round(len(x)/20),callbacks=cblist)
scores=model.evaluate(xy, labels)
print("\n%sL %.2f%%" % (model.metrics_names[1], scores[1]*100))
k = 0
window=15
ww=window*2+1
newxy = np.empty([ww**2,2])
for i in range(ww):
    for j in range(ww):
        # print(i)
        # print(k)
        newxy[[k], ] = [int(i-window)/10, int(j-window)/10]
        k += 1


mgrange = np.arange(ww)/10
xm, ym = np.meshgrid(mgrange, mgrange)
xm=xm-window/10
ym=ym-window/10
fig=plt.figure()
ax1=plt.plot(x1, y1, 'r.', x2, y2, 'b.')
zr=[[0, 0],
    [0, 0]]
ax2=plt.contourf(zr,zr,zr)
def animate(fnumber):
        fnumber+=1
        if fnumber<10:
            fn=str("0"+str(fnumber))
        else:
            fn=str(fnumber)
        wfile=str("weights."+fn+".hdf5")
        model.load_weights(wfile)
        model.compile(loss='binary_crossentropy',
                      optimizer=omt,
                      metrics=['accuracy'])
        prd=model.predict(newxy)
        k = 0
        prdxy = np.empty([ww, ww])
        for i in range(ww):
            for j in range(ww):
                prdxy[[i], [j]] = prd[k]
                k += 1

        ax2=plt.contourf(xm, ym, prdxy)
        return ax2


anim = animation.FuncAnimation(fig, animate , frames = 30, interval=25)
plt.show()


