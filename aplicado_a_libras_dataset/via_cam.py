import cv2, os, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import wisardpkg as wp
import seaborn as sns
from sklearn.metrics import confusion_matrix

train = pd.read_csv("imagens2csv/train.csv")
test = pd.read_csv("imagens2csv/test.csv")

X_train = train.drop(labels=['label'], axis=1).astype('int').values.tolist()
y_train = train['label'].astype('str').values.tolist()

X_test = test.drop(labels=['label'], axis=1).astype('int').values.tolist()
y_test = test['label'].astype('str').values.tolist()





def accuracy(y, y_hat):
  count = 0
  for i in range(len(y)):
    if (y[i] == y_hat[i]):
      count += 1
  return count / len(y)

wsd = wp.Wisard(
  3, # addressSize 5
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

if str(input("deseja salvar o treinamento (modelo_treinado.json)? (yes/no)")) == "yes":
    # salva o treinamento em JSON
    ojsonout = wsd.json()
    # Serializing json
    json_object = json.dumps(ojsonout, indent=4)

    # Writing to sample.json
    with open("modelo_treinado.json", "w") as outfile:
        outfile.write(json_object)


print("Predicting test data...")
pred = wsd.classify(X_test)
print("  - Accuracy on test data: {:.2f}%".format(accuracy(y_test, pred)*100))
cm = confusion_matrix(y_test, pred, labels=[f'{a}' for a in "ABCDEFGILMNOPQRSTUVWY"])
print(cm)



exit(0)
o_img = cv2.imread("O.png", cv2.IMREAD_GRAYSCALE)
y_img = cv2.imread("Y.png", cv2.IMREAD_GRAYSCALE)
o_img = cv2.resize(o_img, (64, 64))
y_img = cv2.resize(y_img, (64, 64))
cv2.imshow("o_img", o_img)
cv2.imshow("y_img", y_img)
flat_o = o_img.flatten()
flat_y = y_img.flatten()
pred = wsd.classify([flat_y,flat_o])
print(pred)
#exit(0)
# define a video capture object
vid = cv2.VideoCapture(0)

while True:
    ret, frame1 = vid.read()
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (64, 64))
    flatten = frame.flatten()
    pred = wsd.classify([flatten])
    #print(flatten)
    # Display the resulting frame
    #if pred[0]['confidence'] != 1.0:
    cv2.putText(frame1, f"{pred[0]}", (150, 150), cv2.FONT_HERSHEY_TRIPLEX, 5, 255)

    cv2.imshow('frame1', frame1)


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    print(pred[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()