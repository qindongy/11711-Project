import numpy as np
import pandas as pd

reviews = pd.read_csv("/Users/anqiluo/Downloads/SetConv-master/AmzSoftwares/amz_review/processedReview.csv")
testIndex = pd.read_csv("/Users/anqiluo/Downloads/SetConv-master/AmzSoftwares/amz_review/softwares_0.6_test_idx.csv")
testReview = reviews.iloc[testIndex.iloc[:,0].values,:]
groudTruth = pd.read_csv("/Users/anqiluo/Downloads/groudTrue.csv",header=None)
pred = pd.read_csv("/Users/anqiluo/Downloads/pre.csv",header=None)
ifCorrect = groudTruth == pred

data = {
    "ifCorrect" : ifCorrect.iloc[:,0].values.tolist(),
    "groudTruth":groudTruth.iloc[:,0].values.tolist(),
    "pred":pred.iloc[:,0].values.tolist(),
    "testReview":testReview.iloc[:,0].values.tolist()
}
df = pd.DataFrame(data)
print(len(data['groudTruth']),len(data['pred']),len(data['testReview']))
