'''
Este algoritmo transforma as imagens 64 X 64 em arquivos CSV

André de J A Ramos
'''

import cv2, os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

os.chdir("../archive")
print(os.getcwd())

root = os.getcwd()
lista_dir = os.listdir(root)

array_geral_train = np.array([[]])

''' 
São 4096 pixels. Esse trecho de código cria um rótulo para as colunas do dataframe
'''
rotulos = np.array([f'pixel{a}' for a in range(0, 4096)])
rotulos = np.insert(rotulos, 0, 'label', axis=0)
print(rotulos)


a = np.array(['A', 'B', 'C', 'D'])
'''

'''
for dir in lista_dir:
    if os.path.isdir(dir):
        print(os.path.join(root, dir))
        if dir == "test":
            train_dir = os.path.join(root, dir)
            letras_dir = os.listdir(train_dir)

            for letra_dir in letras_dir:
                print(letra_dir)
                path_full = os.path.join(root, dir, letra_dir)
                df = pd.DataFrame(None, columns=rotulos)
                if os.path.isdir(path_full):# and not (letra_dir in a):
                    files = os.listdir(path_full)
                    amnt_files = len(files)
                    count = 0

                    for file in files:
                        if os.path.isfile(os.path.join(path_full, file)):
                            count += 1
                            print(os.path.join("archive", dir, letra_dir, file))
                            #os.chdir("..")
                            imagem = cv2.imread(os.path.join("..", "archive", dir, letra_dir, file), cv2.IMREAD_GRAYSCALE)
                            #imgRSZ = cv2.resize(imagem, (10, 10))
                            array = [np.hstack(([letra_dir], imagem.flatten()))]

                            #if(array_geral_train.size == 0):
                                #array_geral_train = array

                            #array_geral_train = np.append(array_geral_train, array, axis=0)


                            df.loc[len(df)] = array[0]
                            #print(df)
                            #df.to_csv("file.csv")
                            #np_array = imgRSZ.flatten()
                            #print(array_geral_train)


                            '''list = ["Hyperion", 27000, "60days", 2000]
                            df.loc[len(df)] = list
                            print(df)'''

                            #cv2.imshow("file", imagem)

                            #os.chdir("archive")

                            #cv2.waitKey(0)
                    df.to_csv(os.path.join("..", "aplicado_a_libras_dataset", "imagens2csv", f"{letra_dir}Tfile.csv"))

#print(df)
#os.chdir("..")
#df.to_csv("file.csv")




