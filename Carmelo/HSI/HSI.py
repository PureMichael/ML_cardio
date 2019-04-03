from scipy import stats
import matplotlib.pyplot as plt
import keras
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# S:/Share/GonzalesC/ML/ML Code/Carmelo/HSI
# random.seed(6)
# keras.set_random_seed(333)
# np.random.seed(856)


df = pd.read_csv('GT_Full_No_0.csv')
bands = np.array(df.columns.values)[1:-1].astype('float')
data = np.array(df.iloc[:, 1:])
data = data.astype('float')
d1 = data[:, 80:]
# d1 = data[:, 1:]

def normalize_data(data):
    try:
        print('trying')
        for i in range(data.shape[0]):
            data[i, :] = data[i, :] / np.max(np.abs(data[i, :]))
    except IndexError:
        data[:] = data[:] / np.max(np.abs(data[:]))
    return data

d1 = normalize_data(d1)
d2= data[:, 0]
data = np.column_stack((d2, d1))

class Element:
    def __init__(self, data_in, ml_type, uniform_sample):
        self.cats, self.cats_elements = np.unique(data_in[:, 0], return_counts=True)
        self.data = []
        self.train_data = []
        self.test_data =[]
        for i in range(len(self.cats)):
            if uniform_sample == False:
                self.data.append(data_in[np.nonzero(data_in[:, 0] == self.cats[i])[0], :])
                order = random.sample(range(self.cats_elements[i]), self.cats_elements[i])
                split_idx = int(np.round(0.75 * self.cats_elements[i]))
            else:
                uniform_number = np.min(self.cats_elements)
                order_uniform = random.sample(range(self.cats_elements[i]), uniform_number)
                # order_uniform=np.arange(uniform_number)
                self.data.append(data_in[np.nonzero(data_in[:, 0] == self.cats[i])[0][order_uniform], :])
                order = random.sample(range(uniform_number), uniform_number)
                split_idx = int(np.round(0.75 * uniform_number))
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
        for i in range(len(self.cats)):
            self.full_train_data = np.concatenate((self.full_train_data, self.train_data[i]))
            self.full_test_data = np.concatenate((self.full_test_data, self.test_data[i]))


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        # clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()

plot_losses = PlotLosses()

def pca_function(data):
    pca = decomposition.PCA(n_components=50)
    pca.fit(data.full_train_data[:, 1:])
    data.pca_train = np.column_stack((data.full_train_data[:, 0], pca.transform(data.full_train_data[:, 1:])))
    pca.fit(data.full_test_data[:, 1:])
    data.pca_test = np.column_stack((data.full_test_data[:, 0], pca.transform(data.full_test_data[:, 1:])))
    return data


def k_means_function(data):
    try:
        print('Try k_means')
        kmeans = KMeans(n_clusters=len(data.cats),
                             random_state=0).fit(data.pca[:, 1:])
        labels = kmeans.labels_
        data.kmeans = np.column_stack((labels, data.pca[:, 1:]))
    except AttributeError:
        print('Except k_means')
        kmeans = KMeans(n_clusters=len(data.cats),
                             random_state=0).fit(data.full_train_data[:, 1:])
        labels = kmeans.labels_
        data.kmeans = np.column_stack((labels, data.full_train_data[:, 1:]))
    return data

def nn_function(train_data,train_labels,test_data,test_labels):
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    num_inputs = train_data.shape[1]-1
    ## might give problems if len(data.cats)!=len(lables)
    num_outputs = len(data.cats)
    model = Sequential()
    model.add(Dense(num_inputs, input_dim=num_inputs,  activation='relu'))
    model.add(Dense(num_inputs * 2, activation='relu'))
    model.add(Dense(num_inputs * 4, activation='relu'))
    model.add(Dense(num_inputs * 8, activation='relu'))
    model.add(Dense(num_inputs * 16, activation='relu'))
    model.add(Dense(num_inputs * 8, activation='relu'))
    model.add(Dense(num_inputs * 4, activation='relu'))
    model.add(Dense(num_inputs * 2, activation='relu'))
    model.add(Dense(num_outputs, activation='softmax'))
    omt = keras.optimizers.Adam(lr=0.0005)
    model.compile(optimizer=omt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    data.history = model.fit(train_data[:, 1:], train_labels,
              epochs = 25,
              verbose = 1,
              batch_size=int(len(train_data[:, 0])/50),
              validation_data=(test_data[:, 1:], test_labels))
    data.prd = model.predict_classes(test_data[:, 1:])
    data.prd_percent = model.predict(test_data[:, 1:])
    scores = model.evaluate(test_data[:, 1:], test_labels)
    print(str(model.metrics_names[1])+' %.2f%%' % (scores[1]*100) + ' accuracy on test data')
    print(np.column_stack((data.prd[0:10], test_data[0:10, 0])))
    output = np.column_stack((data.prd[0:], test_data[0:, 0]))
    return output, data


def plot_n_from_each(data_plot, data_og):
    n = 10
    for i in range(len(data_og.cats)):
        current_set = np.nonzero(data_plot[:, 0] == i)[0]
        plot_these = random.sample(range(len(current_set)), n)
        plt.figure()
        plt.title('Category: '+str(i))
        plt.plot(np.transpose(data_og.full_train_data[current_set[plot_these], 1:]))
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



data = Element(data, '', uniform_sample=True)
data = pca_function(data)


train_data = data.full_train_data[:, 0:]
test_data = data.full_test_data[:, 0:]
order_train = random.sample(range(len(train_data)), len(train_data))
order_test = random.sample(range(len(test_data)), len(test_data))
train_data = train_data[order_train, 0:]
test_data = test_data[order_test, 0:]

train_labels = train_data[:, 0]
test_labels = test_data[:, 0]


output, data = nn_function(train_data,train_labels,test_data,test_labels)



cm = confusion_matrix(test_labels, output[:, 0])
plt.figure()
plot_confusion_matrix(cm,classes=data.cats,
                      title='Confusion matrix')
plt.show()
prd_per = np.max(data.prd_percent, 1)
confident_estimate = np.nonzero(prd_per>0.5)[0]
output2 = output[confident_estimate, :]
right=0
for k in range(len(output2)):
    if output2[k,0]==output2[k,1]:
        right+=1

right/len(output2)
cm2 = confusion_matrix(test_labels[confident_estimate], output2[:, 0])
plt.figure()
plot_confusion_matrix(cm2, classes=data.cats,
                      title='Confusion matrix')
plt.show()
