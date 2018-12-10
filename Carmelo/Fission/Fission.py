import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Input, Activation, Dropout, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, Flatten, Softmax
from keras.models import Model, Sequential

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

nn=6
aa=4
random.seed(nn+aa)
tf.set_random_seed(nn+aa)
np.random.seed(nn+aa)

dir = 'E:/Images_Processed/'
numFiles = (len(os.listdir(dir)))
images = np.empty((numFiles,176,176))
labels = np.ones(numFiles)
for i, filename in enumerate(os.listdir(dir)):
    if np.mod(i,100)==0 or i==numFiles-1:
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
randomize_train = random.sample(range(len(train_imgs)), len(train_imgs))
randomize_test = random.sample(range(len(test_imgs)), len(test_imgs))
train_imgs = train_imgs[randomize_train, :, :]
train_labels = train_labels[randomize_train]
test_imgs = test_imgs[randomize_test, :, :]
test_labels = test_labels[randomize_test]
###
###
train_imgs = train_imgs.astype('float32')/255
train_imgs = np.reshape(train_imgs, (len(train_labels), 176, 176, 1))
train_labels = to_categorical(train_labels)
test_imgs = test_imgs.astype('float32')/255
test_imgs = np.reshape(test_imgs, (len(test_labels), 176, 176, 1))
test_labels = to_categorical(test_labels)
###
###
n=10
random_train_image_idx = random.sample(range(len(train_imgs)),n)
random_test_image_idx = random.sample(range(len(test_imgs)),n)
plt.figure(figsize=(10,4))
plt.suptitle('10 Random Train and Test Images w/ Label')
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(train_imgs[random_train_image_idx[i]].reshape(176,176))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(str(train_labels[random_train_image_idx[i]][1]))
    # ax.set_title(str(train_labels[random_train_image_idx[i]]))
    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(test_imgs[random_test_image_idx[i]].reshape(176,176))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(str(test_labels[random_test_image_idx[i]][1]))
    # ax.set_title(str(test_labels[random_test_image_idx[i]]))

plt.show(block=False)
plt.pause(5)
plt.close()
###
###
print('Creating Model')
model = Sequential()
model.add(Conv2D(8,(3, 3,), input_shape=(176, 176, 1), padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3, 3,), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D((3, 3), strides=3, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3, 3,), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
print('Compiling Model')
sgd = keras.optimizers.SGD(lr=0.01)
model.compile(loss='binary_crossentropy',
                   optimizer=sgd,
              metrics=['accuracy'])
model.summary()
print('Fitting Model')
model.fit(train_imgs, train_labels,
               epochs=10,
               batch_size=234,
               shuffle=True,
               validation_data=(test_imgs, test_labels))
# print('Scoring Model')
# scoresTrain = model.evaluate(train_imgs, train_labels)
# print("\n%sL %.2f%%" % (model.metrics_names[1], scoresTrain[1]*100))
# scoresTest = model.evaluate(test_imgs, test_labels)
# print("\n%sL %.2f%%" % (model.metrics_names[1], scoresTest[1]*100))
predicted_imgs = model.predict_classes(test_imgs,verbose=True)
print(np.column_stack((predicted_imgs, test_labels[:][1])))
predicted_test_image_idx = random.sample(range(len(predicted_imgs)),10)
n=10
plt.figure(figsize=(10,4))
plt.suptitle('Test Image w/ Predicted Label')
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(test_imgs[predicted_test_image_idx[i]].reshape(176,176))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(str(predicted_imgs[i]))

plt.show()

