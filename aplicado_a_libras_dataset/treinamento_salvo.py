'''
Este código faz a classificação a partir de um modelo salvo em json no arquivo via_cam.py
--André de J A Ramos
'''
import json
import pandas as pd
import wisardpkg as wp
from sklearn.metrics import confusion_matrix


test = pd.read_csv("imagens2csv/test.csv")

X_test = test.drop(labels=['label'], axis=1).astype('int').values.tolist()
y_test = test['label'].astype('str').values.tolist()


def accuracy(y, y_hat):
  count = 0
  for i in range(len(y)):
    if (y[i] == y_hat[i]):
      count += 1
  return count / len(y)


# Writing to sample.json
with open("sample.json", "r") as openfile:
    json_object = json.load(openfile)

wsd = wp.Wisard(json_object)



print("Predicting test data...")
pred = wsd.classify(X_test)
print("  - Accuracy on test data: {:.2f}%".format(accuracy(y_test, pred)*100))
cm = confusion_matrix(y_test, pred, labels=[f'{a}' for a in "ABCDEFGILMNOPQRSTUVWY"])
print(cm)
