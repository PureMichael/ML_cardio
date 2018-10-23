import keras
from keras.utils import to_categorical
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, Flatten, Softmax
from keras.models import Model, Sequential

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

dir = 'E:/Images_Processed/'
numFiles = (len(os.listdir(dir)))
images = np.empty((numFiles,176,176))
labels = np.ones(numFiles)
for i, filename in enumerate(os.listdir(dir)):
    if np.mod(i,100)==0:
        print(np.round(100*i/numFiles,1))
    images[i,:,:]=mpimg.imread(dir+filename)
    if filename[0]=='n':
        labels[i]=0

###
cat_no = np.nonzero(labels==0)
no_labels = labels[cat_no[0][:]]
no_img = images[cat_no[0][:], :, :]
yes_labels = labels[cat_no[0][-1]+1:]
yes_img = images[cat_no[0][-1]+1:, :, :]
no_idx = random.sample(range(len(no_labels)), len(no_labels))
yes_idx = random.sample(range(len(yes_labels)), len(no_labels))
trainNum = np.round(0.75*len(cat_no[0][:]))
train_no_idx = no_idx[:int(trainNum)]
train_yes_idx = yes_idx[:int(trainNum)]
test_no_idx = no_idx[int(trainNum):]
test_yes_idx = no_idx[int(trainNum):]
###
###
no_train_img = no_img[train_no_idx, :, :]
yes_train_img = yes_img[train_yes_idx, :, :]
no_train_labels = no_labels[train_no_idx]
yes_train_labels = yes_labels[train_yes_idx]
###
###
no_test_img = no_img[test_no_idx, :, :]
yes_test_img = yes_img[test_yes_idx, :, :]
no_test_labels = no_labels[test_no_idx]
yes_test_labels = yes_labels[test_yes_idx]
###
###
train_imgs = np.append(yes_train_img, no_train_img, 0)
train_labels = np.append(yes_train_labels, no_train_labels)
test_imgs = np.append(yes_test_img, no_test_img, 0)
test_labels = np.append(yes_test_labels, no_test_labels)
###
###
train_imgs = train_imgs.astype('float32')/255
train_imgs = np.reshape(train_imgs, (len(train_labels), 176, 176, 1))
train_labels = to_categorical(train_labels)
test_imgs = test_imgs.astype('float32')/255
test_imgs = np.reshape(test_imgs, (len(test_labels), 176, 176, 1))
test_labels = to_categorical(test_labels)
###
print('Creating Model')
model = Sequential()
model.add(Conv2D(8,(3, 3,), activation='relu', input_shape=(176, 176, 1), padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D((2, 2), strides=3, padding='same'))
model.add(Conv2D(32,(3, 3,), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D((3, 3), strides=3, padding='same'))
model.add(Conv2D(32,(3, 3,), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.add(Softmax())
print('Compiling Model')
sgd = keras.optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy',
                   optimizer='sgd')
print('Fitting Model')
model.fit(train_imgs, train_labels,
               epochs=10,
               batch_size=128,
               shuffle=True,
               validation_data=(test_imgs, test_labels))
print('Scoring Model')
scoresTrain = model.evaluate(train_imgs, train_labels)
print("\n%sL %.2f%%" % (model.metrics_names[1], scoresTrain[1]*100))
scoresTest = model.evaluate(test_imgs, test_labels)
print("\n%sL %.2f%%" % (model.metrics_names[1], scoresTest[1]*100))

