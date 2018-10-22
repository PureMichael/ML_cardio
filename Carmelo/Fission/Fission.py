import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import keras
from keras.utils import to_categorical
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, Flatten, Softmax
from keras.models import Model, Sequential

# images = mpimg.imread('E:/Fission/Type1_Trace/17-32/121025 72011-98_X00945_Y00780_ID00986.png')
# plt.imshow(images, cmap='gray')
# plt.show()
# images = Image.open('E:/Fission/Type1_Trace/17-32/121025 72011-98_X00945_Y00780_ID00986.png')
# images = Image.open('C:/Users/gonzalesc/Desktop/72011-94_X00086_Y01803_ID00795.jpg')
# print(images)
# images.show()


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


cat_no = np.nonzero(labels>0)



trainNum = np.round(0.75*len(cat_no[0][:]))




labels = to_categorical(labels)
print(labels)
images = images.astype('float32')/255
print(len(images))
images = np.reshape(images, (len(labels), 176, 176, 1))


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
sgd = keras.optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy',
                   optimizer='sgd')

model.fit(images, labels,
               epochs=10,
               shuffle=True)


