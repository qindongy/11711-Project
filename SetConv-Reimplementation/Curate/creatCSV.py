import numpy as np
import pandas as pd

allOut = pd.read_csv("./allOut.csv")
labels = pd.read_csv("./labels.csv")
print(allOut.shape)
print(len(labels))
labels = labels.to_numpy().squeeze(1)
positive = np.where(labels==1.0)[0]
negative = np.where(labels==0.0)[0]
sizePositive = len(positive)
sizeNegative = len(negative)
print("negative is the minority class ",sizePositive > sizeNegative)
imbalance_ratio = 0.6
sizeNegative = int(sizePositive * imbalance_ratio)
indexNegative = np.arange(sizeNegative).astype(int)
np.random.shuffle(indexNegative)
negative = negative[indexNegative]
labels = np.append(labels[negative],labels[positive])
allOutNeg = allOut.iloc[negative]
allOutPos = allOut.iloc[positive]
allOut = pd.concat([allOutNeg,allOutPos],axis=0)
with open("/Users/anqiluo/Downloads/softwares.txt",'w') as out:
    for i in range(len(allOut)):
        line = allOut.iloc[i].values[0]
        out.write(line)
        out.write("\n")
out.close()
print(len(allOut))

train_ratio = 0.6
valid_ratio = 0.1
index = np.arange(len(allOut))
np.random.shuffle(index)
train_index = index[:int(len(index)*train_ratio)]
valid_index = index[int(len(index)*train_ratio):int(len(index)*(train_ratio+valid_ratio))]
test_index = index[int(len(index)*(train_ratio+valid_ratio)):]

# print class distribution
from collections import Counter
print(Counter(labels[train_index]))
print(Counter(labels[valid_index]))
print(Counter(labels[test_index]))
pd.DataFrame(train_index).to_csv('./softwares_{}_train_idx.csv'.format(train_ratio), header=None, index=False)
pd.DataFrame(valid_index).to_csv('./softwares_{}_valid_idx.csv'.format(train_ratio), header=None, index=False)
pd.DataFrame(test_index).to_csv('./softwares_{}_test_idx.csv'.format(train_ratio), header=None, index=False)
