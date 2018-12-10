import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
import pandas as pd
plt.rcParams['animation.ffmpeg_path']='C:/Program Files/ffmpeg/bin/ffmpeg.exe'
import os
path=os.path.dirname(os.path.realpath('circleml.py'))
print(path)
# random.seed(6)
# tf.set_random_seed(333)
# np.random.seed(856)
loss='binary_crossentropy'
loadData=0
circleData=0
saveVideo=0
showPlots=1

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

if loadData == 0 :
    numm=(2*np.pi)/500
    t = np.empty(round(2*np.pi/numm)+1)
    j = 0
    for i in frange(0,2*np.pi,numm):
        t[j]=i
        j+=1
    rlen = len(t)
    rnd = 20
    xr1 = (np.random.rand(rlen) - 0.5) / rnd
    yr1 = (np.random.rand(rlen) - 0.5) / rnd
    labels1 = np.zeros(len(t))
    xr2 = (np.random.rand(rlen) - 0.5) / rnd
    yr2 = (np.random.rand(rlen) - 0.5) / rnd
    labels2 = np.ones(len(t))
    labelsn = np.concatenate((labels1, labels2))

    if circleData ==1 :
        x1 = np.add((0.5 * np.cos(t)), xr1)
        y1 = np.add(0.5 * np.sin(t), yr1)
        x2 = np.add((np.cos(t)), xr2)
        y2 = np.add(np.sin(t), yr2)
    else :
        rn=1
        rd=1
        thetam=1
        x1 = np.add((rn / (t+rd)) * np.cos(thetam*t), xr1)
        y1 = np.add((rn / (t + rd)) * np.sin(thetam * t), yr1)
        x2 = np.add((-rn / (t + rd)) * np.cos(-1*thetam * t), xr2)
        y2 = np.add((-rn / (t + rd)) * np.sin(thetam * t), yr2)

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    xyn = np.column_stack((x, y))
    print(labelsn.shape)
    order = random.sample(range(xyn.shape[0]), xyn.shape[0])
    xy = np.empty(shape=(int(xyn.shape[0]), int(xyn.shape[1])))
    labels=np.empty(shape=(int(labelsn.shape[0])))
    for ii in order:
        xy[ii, :] = xyn[order[ii], :]
        labels[ii] = labelsn[order[ii]]
    print(type(xy[1,1]))
    print(np.column_stack((xy,labels)))
else :
    xl = pd.ExcelFile('swirl_chub.xls')
    mldata = xl.parse('Sheet1')
    print(mldata.shape)
    split = round(0.95*int(mldata.shape[0]))
    order = np.arange(mldata.shape[0]-1)
    data = np.empty(shape=(int(mldata.shape[0]), int(mldata.shape[1])))
    for ii in order:
        data[ii, :] = mldata.iloc[order[ii], 0:]
    xy = data[:,0:-1]
    x1 = xy[:int(xy.shape[0] / 2), 0]
    y1 = xy[:int(xy.shape[0] / 2), 1]
    x2 = xy[int(xy.shape[0] / 2):, 0]
    y2 = xy[int(xy.shape[0] / 2):, 1]
    del data
    del xy
    order = random.sample(range(mldata.shape[0]), mldata.shape[0])
    data = np.empty(shape=(int(mldata.shape[0]), int(mldata.shape[1])))
    for ii, idx in enumerate(order):
        data[ii, :] = mldata.iloc[order[ii], 0:]
    xy = data[:,0:-1]
    labels = data[:,-1]
split = round(0.75*int(xy.shape[0]))
xytrain = xy[:split,]
labelstrain = labels[:split]
xytest = xy[split:,]
labelstest=labels[split:]
if showPlots ==1 :
    plt.figure()
    for i in range(len(labelstrain)):
        if labelstrain[i]==0:
            plt.plot(xytrain[i,0],xytrain[i,1],'b.')
        else:
            plt.plot(xytrain[i, 0], xytrain[i, 1], 'r.')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    plt.figure()
    for i in range(len(labelstest)):
        if labelstest[i]==0:
            plt.plot(xytest[i,0],xytest[i,1],'b.')
        else:
            plt.plot(xytest[i, 0], xytest[i, 1], 'r.')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

model = keras.Sequential()
model.add(keras.layers.Dense(4, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
cblist=[keras.callbacks.ModelCheckpoint('weights.{epoch:02d}.hdf5', save_weights_only=True, period=1)]
omt = keras.optimizers.Adam(lr=0.01)
model.compile(loss=loss,
              optimizer=omt,
              metrics=['accuracy'])
epic=75

print(labelstrain)
model.fit(xytrain, labelstrain, epochs=epic, verbose=1, batch_size=round(len(labels)/20),callbacks=cblist)
scoresTrain=model.evaluate(xytrain, labelstrain)
print("\n%sL %.2f%%" % (model.metrics_names[1], scoresTrain[1]*100))
scoresTest=model.evaluate(xytest, labelstest)
print("\n%sL %.2f%%" % (model.metrics_names[1], scoresTest[1]*100))




k = 0
window=1+int(np.round(10*np.max(xy)))
ww=window*2+1
newxy = np.empty([ww**2,2])
for i in range(ww):
    for j in range(ww):
        newxy[[k], ] = [int(i-window)/10, int(j-window)/10]
        k += 1
mgrange = np.arange(ww)/10
xm, ym = np.meshgrid(mgrange, mgrange)
xm = xm-window/10
ym = ym-window/10
fig = plt.figure()
ax1 = plt.plot(x1, y1, 'r.', x2, y2, 'b.')
zr = np.eye(2)
ax2 = plt.contourf(0*zr,0*zr,zr,10, cmap=plt.cm.viridis)
cb = plt.colorbar(ax2,ticks=np.round(np.arange(11)/10, 1))
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
        prd = model.predict(newxy)
        k = 0
        prdxy = np.empty([ww, ww])
        for j in range(ww):
            for i in range(ww):
                prdxy[[i], [j]] = prd[k]
                k += 1
        ax2=plt.contourf(xm, ym, prdxy)
        plt.title(loss + ' - Epoch: '+str(fnumber))
    else:
        ax2=fig
        plt.title(loss + ' - Epoch: ' + str(epic))
    return ax2
anim = animation.FuncAnimation(fig, animate, frames=epic+25, interval=100, repeat=False)
if saveVideo == 1:
    anim.save('learning_video_spiral.mp4',writer=writer,dpi=dpi)
plt.show()

