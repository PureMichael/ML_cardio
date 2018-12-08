from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


df = pd.read_csv('GT_Test.csv')
data = np.zeros((df.shape[0],df.shape[1]-1))
for i in range(len(df)):
    data[i, :] = df.iloc[i, 1:]

og_data = data

pca = decomposition.PCA(n_components=50)
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
    plot_these = random.sample(range(len(d[i].data)), 10)
    plt.figure(11+i)
    plt.plot(np.transpose(d[i].data[np.array(plot_these), 1:]))
    plt.figure(10)
    plt.plot(np.transpose(np.mean(d[i].data[:,1:],0)))

plt.show()
