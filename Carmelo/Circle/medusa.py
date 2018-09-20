import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd
plt.rcParams['animation.ffmpeg_path']='C:/Program Files/ffmpeg/bin/ffmpeg.exe'
import os
path=os.path.dirname(os.path.realpath('circleml.py'))
print(path)
saveVideo=0
loss='categorical_crossentropy'
numPts=100
t = np.arange(round(numPts*np.pi/2),round(4*np.pi*numPts))
t = t/numPts
rlen = len(t)
rnd = 5
sf=(1/(2*np.pi))
sf2=2
iter = np.arange(8)
threads = np.pi/4 * iter
xy=np.empty((0,2))
labels=np.array([])
for ii, theta in enumerate(threads):
    xrandom = (np.random.rand(rlen)-0.5)/rnd
    yrandom = (np.random.rand(rlen)-0.5)/rnd
    x = sf*(t*np.cos(theta) - sf2*np.sin(t)*np.sin(theta))
    y = sf*(t*np.sin(theta) + sf2*np.sin(t)*np.cos(theta))
    x = np.add(x,xrandom)
    y= np.add(y,yrandom)
    plt.plot(x,y,'.')
    label = ii*np.ones((len(x),1))
    data = np.column_stack((x, y))
    xy = np.concatenate((xy, data))
    labels = np.append(labels, label)
plt.show(block=False)
plt.pause(2)
plt.close()
xyl = np.column_stack((xy,labels))
order = random.sample(range(xyl.shape[0]),xyl.shape[0])
data = np.zeros((xyl.shape[0],xyl.shape[1]))
data[:]=np.nan
for ii, jj in enumerate(order):
    data[ii,:] = xyl[jj,:]
split = round(0.75*data.shape[0])
xtrain = data[:split, 0]
ytrain = data[:split, 1]
xytrain=np.column_stack((xtrain,ytrain))
labelstrain = data[:split, -1]
encoder = LabelEncoder()
encoder.fit(labelstrain)
encoded_y=encoder.transform(labelstrain)
hot_labelstrain=keras.utils.to_categorical(encoded_y)
xtest = data[split:, 0]
ytest = data[split:, 1]
xytest=np.column_stack((xtest,ytest))
labelstest = data[split:, -1]
encoder = LabelEncoder()
encoder.fit(labelstest)
encoded_y=encoder.transform(labelstest)
hot_labelstest=keras.utils.to_categorical(encoded_y)
model = keras.Sequential()
model.add(keras.layers.Dense(4,input_dim=2, activation='relu'))
# model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
# model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dense(32, activation='relu'))
# model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
# model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(8, activation='sigmoid'))
omt = keras.optimizers.Adam(lr=0.00075)
epic=50
cblist=[keras.callbacks.ModelCheckpoint('weights.{epoch:02d}.hdf5', save_weights_only=True, period=1)]
model.compile(optimizer=omt,
              loss=loss,
              metrics=['accuracy'])
model.fit(xytrain, hot_labelstrain, epochs=epic, verbose=1, batch_size=100,callbacks=cblist)
scores = model.evaluate(xytrain, hot_labelstrain,verbose=0)
prd = model.predict_classes(xytest)
print(np.column_stack((prd, labelstest)))
scoresTest = model.evaluate(xytest, hot_labelstest,verbose=0)
print(str(model.metrics_names[1])+' %.2f%%' % (scores[1]*100) + ' accuracy on trained data')
print(str(model.metrics_names[1])+' %.2f%%' % (scoresTest[1]*100) + ' accuracy on test data')
k = 0
cten=15
window=1+int(np.round(cten*np.max(xy)))
ww=window*2+1
newxy = np.empty([ww**2,2])
for i in range(ww):
    for j in range(ww):
        newxy[[k], ] = [int(i-window)/cten, int(j-window)/cten]
        k += 1
mgrange = np.arange(ww)/cten
xm, ym = np.meshgrid(mgrange, mgrange)
xm = xm-window/cten
ym = ym-window/cten
fig = plt.figure()
ax1 = plt.plot(xyl[:,0],xyl[:,1],'k.')
zr = 8*np.eye(2)
tn=8
ax2 = plt.contourf(0*zr,0*zr,zr,8, cmap=plt.cm.jet)
cb = plt.colorbar(ax2)
cb.set_ticks((np.arange(tn)+0.5))
cb.set_ticklabels(np.arange(tn))
dpi = 100
Writer = animation.writers['ffmpeg']
writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
def animate(fnumber):
    fnumber+=1
    if fnumber<=epic:
        if fnumber<10:
            fn=str("0"+str(fnumber))
        else:
            fn=str(fnumber)
        wfile=str("weights."+fn+".hdf5")
        model.load_weights(wfile)
        model.compile(loss=loss,
                      optimizer=omt,
                      metrics=['accuracy'])
        prd = model.predict_classes(newxy)
        k = 0
        prdxy = np.empty([ww, ww])
        for j in range(ww):
            for i in range(ww):
                prdxy[[i], [j]] = prd[k]
                k += 1
        ax2=plt.contourf(xm, ym, prdxy,8, cmap=plt.cm.jet)

        plt.title(loss + ' - Epoch: '+str(fnumber))
    else:
        ax2=fig
        plt.title(loss + ' - Epoch: ' + str(epic))
    return ax2
anim = animation.FuncAnimation(fig, animate, frames=epic+25, interval=100, repeat=False)
if saveVideo == 1:
    anim.save('medusa.mp4',writer=writer,dpi=dpi)
plt.show()
