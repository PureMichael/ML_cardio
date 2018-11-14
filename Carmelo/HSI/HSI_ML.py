import pandas as pd
import numpy as np
import random


df = pd.read_csv('GT_Test.csv')
h = np.histogram(df.iloc[:, 1], bins=np.arange(0, 8))

class Element:
    def __init__(self, data, ii):
        self.data= data.iloc[np.nonzero(data.iloc[:, 1] == ii)[0], :]
        order = random.sample(range(len(self.data)),len(self.data))
        split_idx = int(np.round(0.75*len(self.data)))
        train_set = order[:split_idx]
        test_set = order[split_idx:]
        self.train_data = self.data.iloc[train_set, :]
        self.test_data = self.data.iloc[test_set, :]
        # self.train_labels = self.data.iloc[train_set, 1]
        # self.test_labels = self.data.iloc[test_set, 1]

data=[]
for i in np.unique(df.iloc[:, 1]):
    data.append(Element(df, int(i)))

train_data = pd.DataFrame()
test_data = pd.DataFrame()
# train_labels = pd.DataFrame()
# test_labels = pd.DataFrame()

for i in np.arange(len(data)):
    train_data = train_data.append(data[i].train_data)
    test_data = test_data.append(data[i].test_data)
    # train_labels = train_labels.append(data[i].train_labels)
    # test_labels = test_labels.append(data[i].test_labels)

order_train = random.sample(range(len(train_data)), len(train_data))
order_test = random.sample(range(len(test_data)), len(test_data))

train_data = train_data.iloc[order_train, :]
test_data = test_data.iloc[order_test, :]
train_labels = train_data.iloc[:, 1]
test_labels = test_data.iloc[:, 1]


