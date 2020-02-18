# -*- coding: utf-8 -*-

from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
names = os.listdir('C:/Users/Pichau/Desktop/TCC/freesound-audio-tagging/patchsTrain')
imagens_treino = []

print("Carregando imagens")
for i in names[0:3000]:
    arr = Image.open(r"../freesound-audio-tagging/patchsTrain/" + i)
    arr = np.array(arr)
    imagens_treino.append(arr)
imagens_treino = np.asarray(imagens_treino)
imagens_treino = np.mean(imagens_treino, axis=-1, keepdims=True)
print(imagens_treino.shape)


print("Carregando classes")
identificacoes_treino = np.loadtxt("../freesound-audio-tagging/categorias.txt", delimiter='\n', dtype= 'str')
categorias = sorted(set(identificacoes_treino))
dic = dict()
for n, f in enumerate(categorias):
    dic[f] = n
for i in range(0, len(identificacoes_treino)):
    identificacoes_treino[i] = dic[identificacoes_treino[i]]
identificacoes_treino = to_categorical(identificacoes_treino)
identificacoes_treino = identificacoes_treino[0:3000]

print("Inicializando modelo")
modelo = Sequential()
modelo.add(Conv2D(100, (2, 2), input_shape=(128, 8, 1), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Conv2D(50, (2, 2), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Flatten())
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(41, activation='softmax', name='predict'))
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

historico = modelo.fit(imagens_treino, identificacoes_treino, epochs=5, validation_split=0.2)
