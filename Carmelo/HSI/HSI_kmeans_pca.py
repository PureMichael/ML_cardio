from sklearn.cluster import KMeans
from sklearn import decomposition
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


df = pd.read_csv('GT_Test.csv')
data = np.zeros((df.shape[0], df.shape[1]-1))
for i in range(len(df)):
    data[i, :] = df.iloc[i, 1:]

cats = np.unique(data[:, 0])
min_data = np.array(100000)
for i in cats:
    l = len(np.nonzero(data[:,0]==i)[0])
    min_data = np.column_stack((min_data, l))

min_data = np.min(min_data)
og_data = data
new_data=np.zeros((min_data*len(cats)+len(cats)-1, data.shape[1]))
for i in cats:
    print(i)
    new_set = np.nonzero(data[:,0]==i)[0]
    order = random.sample(range(len(new_set)), min_data)
    print(int(min_data*(i)+i))
    print(int((i+1)*min_data+i))
    new_data[int(min_data*(i)+i):int((i+1)*min_data+i)] = (data[new_set[order], :])

data = np.array(new_data)
plt.plot(data[:,0])
plt.show()


pca = decomposition.PCA(n_components=25)
pca.fit(data[:, 1:])
data = np.column_stack((data[:, 0], pca.transform(data[:, 1:])))

kmeans = KMeans(n_clusters=7,
                random_state=0).fit(data[:, 1:])
labels = kmeans.labels_


class Cats:
    def __init__(self, data, labels, ii):
        self.data = data[np.nonzero(labels == ii)[0], :]


d = []
for i in range(len(np.unique(og_data[:, 0]))):
    print(i)
    d.append(Cats(og_data, labels, i))
    plot_these = random.sample(range(len(d[i].data)), 20)
    plt.figure(1+i)
    plt.title('category: '+str(i))
    plt.plot(np.transpose(d[i].data[np.array(plot_these), 1:]))
    plt.figure(0)
    plt.plot(np.transpose(np.mean(d[i].data[:,1:],0)))

plt.show()

l_map=np.array([])
for i in range(7):
    c_us = og_data[np.nonzero(labels==i)[0], 0]
    np.concatenate((l_map, stats.mode(c_us)[0]))




































