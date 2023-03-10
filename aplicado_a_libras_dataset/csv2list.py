import cv2, os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import wisardpkg as wp

train = pd.read_csv("imagens2csv/train.csv")

#print(train)
test = pd.read_csv("imagens2csv/test.csv")
df = train.drop(labels=['label'], axis=1)


X_train = df.values.tolist()
y_train = train['label'].values.astype('str').tolist()
X_test = test.values.tolist()


def accuracy(y, y_hat):
  count = 0
  for i in range(len(y)):
    if (y[i] == y_hat[i]):
      count += 1
  return count / len(y)

wsd = wp.Wisard(
  5, # addressSize
  bleachingActivated=True,
  ignoreZero=False,
  completeAddressing=True,
  verbose=True,
  indexes=[],
  base=256,
  confidence=1,
  returnActivationDegree=False,
  returnConfidence=False,
  returnClassesDegrees=False
)

print("Training...")
wsd.train(X_train, y_train)

print("Predicting train data...")
pred = wsd.classify(X_train)
print("  - Accuracy on train data: {:.2f}%".format(accuracy(y_train, pred)*100))