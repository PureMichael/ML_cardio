from scipy import stats
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
import keras

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans

df = pd.read_csv('GT_Test.csv')
data = np.array(df.iloc[:, 1:])

class Element:
    def __init__(self, data_in, ml_type):
        self.cats, self.cats_elements = np.unique(data_in[:, 0], return_counts=True)
        self.data = []
        self.train_data = []
        self.test_data =[]
        for i in self.cats:
            self.data.append(data_in[np.nonzero(data_in[:, 0] == self.cats[i])[0], :])
            order = random.sample(range(self.cats_elements[i]), self.cats_elements[i])
            split_idx = int(np.round(0.75 * self.cats_elements[i]))
            train_set = order[:split_idx]
            test_set = order[split_idx:]
            self.train_data.append(self.data[i][train_set, :])
            self.test_data.append(self.data[i][test_set, :])
        if ml_type =='NN' or ml_type =='':
            self.ml_type = ml_type
            self.neural_network()
    def neural_network(self):
        self.full_train_data = np.zeros((0,self.data[0].shape[1]))
        self.full_test_data = np.zeros((0, self.data[0].shape[1]))
        for i in self.cats:
            self.full_train_data = np.concatenate((self.full_train_data, self.train_data[i]))
            self.full_test_data = np.concatenate((self.full_test_data, self.test_data[i]))


def pca_function(data):
    pca = decomposition.PCA(n_components=25)
    pca.fit(data.full_train_data[:, 1:])
    data.pca = np.column_stack((data.full_train_data[:, 0], pca.transform(data.full_train_data[:, 1:])))
    return data


def k_means_function(data):
    try:
        print('Try')
        kmeans = KMeans(n_clusters=len(data.cats),
                             random_state=0).fit(data.pca[:, 1:])
        labels = kmeans.labels_
        data.kmeans = np.column_stack((labels, data.pca[:, 1:]))
    except AttributeError:
        print('Except')
        kmeans = KMeans(n_clusters=len(data.cats),
                             random_state=0).fit(data.full_train_data[:, 1:])
        labels = kmeans.labels_
        data.kmeans = np.column_stack((labels, data.full_train_data[:, 1:]))
    return data

def nn_function(data):
    order_train = random.sample(range(len(data.full_train_data)), len(data.full_train_data))
    order_test = random.sample(range(len(data.full_test_data)), len(data.full_test_data))
    train_data = data.full_train_data[order_train, :]
    test_data = data.full_test_data[order_test, :]
    train_labels = to_categorical(train_data[:,0])
    test_labels = to_categorical(test_data[:, 0])
    num_inputs = train_data.shape[1]-1
    num_outputs = len(data.cats)
    model = Sequential()
    model.add(Dense(num_inputs, input_dim=num_inputs,  activation='relu'))
    model.add(Dense(num_outputs, activation='softmax'))
    omt = keras.optimizers.Adam(lr=0.00001)
    model.compile(optimizer=omt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data[:, 1:], train_labels,
              epochs =25,
              verbose=1,
              batch_size=int(len(test_data[:, 0])/50))
    prd = model.predict_classes(test_data[:,1:])
    scores = model.evaluate(test_data[:, 1:], test_labels)
    print(str(model.metrics_names[1])+' %.2f%%' % (scores[1]*100) + ' accuracy on test data')
    print(np.column_stack((prd[0:10], test_data[0:10, 0])))


def plot_n_from_each(data_plot, data_og):
    n = 10
    for i in range(len(data_og.cats)):
        current_set = np.nonzero(data_plot[:, 0] == i)[0]
        plot_these = random.sample(range(len(current_set)), n)
        plt.figure()
        plt.title('Category: '+str(i))
        plt.plot(np.transpose(data_og.full_train_data[current_set[plot_these], 1:]))
    plt.show()


data = Element(data, '')
nn_function(data)
# data = pca_function(data)
# data = k_means_function(data)
# plot_n_from_each(data.kmeans, data)



# data.full_train_data = np.random.shuffle(data.full_train_data)




