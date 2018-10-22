import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import keras
# from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU
# from keras.models import Model


image = mpimg.imread('E:/Fission/Type1_Trace/17-32/121025 72011-98_X00945_Y00780_ID00986.png')
# image = Image.open('E:/Fission/Type1_Trace/17-32/121025 72011-98_X00945_Y00780_ID00986.png')
# image = Image.open('C:/Users/gonzalesc/Desktop/72011-94_X00086_Y01803_ID00795.jpg')
# image.show()

# print(type(image))
plt.imshow(image, cmap='gray')
plt.show(block=False)
image[:,:,1]=image
print(image.shape())

image =  image.resize((100,100), PIL.Image.BICUBIC)


# image = np.reshape(image, (7*177,7*177))
plt.imshow(image, cmap='gray')
plt.show()



# input_img = Input(shape=(176, 176, 1))
# x = Conv2D(8,(3, 3,), activation='relu', padding='same')(input_img)
# x = BatchNormalization(axis=1)(x)
# # x = ReLU()(x)
# x = MaxPooling2D((2, 2), strides=3, padding='same')(x)
# x = Conv2D(32,(3, 3,), activation='relu', padding='same')(x)
# x = BatchNormalization(axis=1)(x)
# # x = ReLU()(x)
# x = Dense(2, input_shape=(x.shape[0],x.shape[1]), activation='softmax')
# classifier = Model(input_img, x)
# sgd = keras.optimizers.SGD(lr=0.01)
# classifier.compile(loss='crossentropy',
#                    optimizer='sgd')
#
# classifier.fit(images, labels,
#                epochs=10,
#                shuffle=True,
#                validation_data=(test_images, test_labels))


