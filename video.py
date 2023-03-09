'''
Este algoritmo exibe, em janela, uma imagem do sinal de libras do dataset da pasta archive.
E modifica a imagem para ser exibida em preto e branco.

André de J A Ramos
'''

import cv2, os
import numpy as np
from matplotlib import pyplot as plt

os.chdir("archive")
print(os.getcwd())

root = os.getcwd()
lista_dir = os.listdir(root)

for dir in lista_dir:
    if os.path.isdir(dir):
        print(os.path.join(root, dir))
        if dir == "train":
            train_dir = os.path.join(root, dir)
            letras_dir = os.listdir(train_dir)

            for letra_dir in letras_dir:
                print(letra_dir)
                path_full = os.path.join(root, dir, letra_dir)
                if os.path.isdir(path_full):
                    for file in os.listdir(path_full):
                        if os.path.isfile(os.path.join(path_full, file)):
                            print(os.path.join("archive", dir, letra_dir, file))
                            os.chdir("..")
                            imagem = cv2.imread(os.path.join("archive", dir, letra_dir, file), cv2.IMREAD_GRAYSCALE)
                            #hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

                            limiar = 127
                            maximo = 255

                            ret, thresh2 = cv2.threshold(imagem, limiar, maximo, cv2.THRESH_BINARY)
                            '''
                            a binarização aqui funciona melhor com threshold adaptativo
                             conforme consta na documentação do opencv 
                             https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
                            '''
                            th2 = cv2.adaptiveThreshold(imagem, 255,
                                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 11, 2)
                            cv2.imshow("file", th2)
                            os.chdir("archive")
                            cv2.waitKey(0)



