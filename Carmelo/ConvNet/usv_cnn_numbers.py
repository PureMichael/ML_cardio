import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

"""
# simple autoencoder
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats, compression factor of 24.5 assuming input is 784 floats
# this is our input placeholder
input_img = Input(shape=(784, ))
# encoded is the encoded represetation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
                # activity_regularizer=regularizers.l2(1e-5)
# decoded is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input ot its reconstruction
autoencoder = Model(input_img, decoded)
# this modelmaps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for encoded (32-dim) input
encoded_input = Input(shape=(encoding_dim, ))
# retrieve the last layer of the autoencodre model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# opt = keras.optimizers.Adadelta(lr=0.01)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
(xtrain, _), (xtest, _) = mnist.load_data()
xtrain = xtrain.astype('float32')/255
xtest = xtest.astype('float32')/255
xtrain = xtrain.reshape((len(xtrain), np.prod(xtrain.shape[1:])))
xtest = xtest.reshape((len(xtest), np.prod(xtest.shape[1:])))
print(xtrain.shape)
print(xtest.shape)
autoencoder.fit(xtrain, xtrain,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(xtest, xtest))
"""
"""
# deep autoencoder
encoding_dim = 32
(xtrain, _), (xtest, _) = mnist.load_data()
xtrain = xtrain.astype('float32')/255
xtest = xtest.astype('float32')/255
xtrain = xtrain.reshape((len(xtrain), np.prod(xtrain.shape[1:])))
xtest = xtest.reshape((len(xtest), np.prod(xtest.shape[1:])))

input_img = Input(shape=(784, ))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

autoencoder.fit(xtrain, xtrain,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(xtest,xtest))

encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim, ))
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

# encdoe and decode some digitis
# these are from the test batch
encoded_imgs = encoder.predict(xtest)
decoded_imgs = decoder.predict(encoded_imgs)
n = 10
plt.figure(figsize=(15, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(xtest[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconsturciton
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
"""

input_img = Input(shape=(28, 28, 1))
x = Conv2D(16,(3, 3,), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8,(3, 3,), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8,(3, 3,), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# representation at this point is 4, 4, 8

x = Conv2D(8,(3, 3,), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8,(3, 3,), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16,(3, 3,), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1,(3, 3,), activation='sigmoid', padding='same')(x)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(xtrain, xlab), (xtest, _) = mnist.load_data()
print(xtrain.shape)
print(type(xtrain))
xtrain = xtrain.astype('float32')/255
xtest = xtest.astype('float32')/255
xtrain = np.reshape(xtrain, (len(xtrain), 28, 28,1))
xtest = np.reshape(xtest, (len(xtest), 28, 28, 1))
#
# tensorboars --logdir=/tmp/autoencoder

autoencoder.fit(xtrain,xtrain,
                epochs=1,
                batch_size=128*2,
                shuffle=True,
                validation_data=(xtest, xtest))
# callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]

encoded_imgs = encoder.predict(xtest)

decoded_imgs = autoencoder.predict(xtest)
print(decoded_imgs)



n=10
plt.figure(figsize=(15,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(xtest[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show(block=False)

n=10
plt.figure(figsize=(15,8))
for i in range(n):
    ax=plt.subplot(1,n,i+1)
    plt.imshow(encoded_imgs[i].reshape(4,4*8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()






# with open('model.json' ,'w') as file:
#     file.write(autoencoder.to_json())
# autoencoder.save_weights('weights.h5')
#


