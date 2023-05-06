from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
#from keras.preprocessing import image
import keras.utils as image
import numpy as np

#import train data
train_datagen = ImageDataGenerator(shear_range=0.1,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rotation_range=50,
                                   brightness_range=(0.25, 1.3))


train_data = train_datagen.flow_from_directory('../waste_dataset/TRAIN',
                                                  color_mode="grayscale",
                                                 target_size=(250, 250),
                                                 class_mode='sparse',
                                                 shuffle=True,
                                               seed=1)
#import test data

test_datagen = ImageDataGenerator()
test_data = test_datagen.flow_from_directory("../waste_dataset/TEST",
                                                            color_mode="grayscale",
                                                           batch_size=32,
                                                           target_size=(250, 250),
                                                           class_mode='sparse',
                                                           shuffle=True,seed=1)

classes= ["Organic", "Recycle"]


for x, y in test_data:
    print(len(x))
    for imagem in x:
        img_linha = np.ravel(imagem)
        print(img_linha)
        new_img = img_linha.reshape(250, 250)
        #img = image.array_to_img(new_img)
        plt.imshow(new_img)
        plt.show()
    break


new_img = image.load_img("../waste_dataset/TEST/R/R_10012.jpg", target_size=(250, 250))

#img = np.expand_dims(img, axis=0)

plt.imshow(new_img)
plt.show()