# -*- coding: utf-8 -*-

import tensorflow
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
for i in names[0:46395]:
    arr = Image.open(r"../freesound-audio-tagging/patchsTrain/" + i)
    arr = np.array(arr)
    imagens_treino.append(arr)
imagens_treino = np.asarray(imagens_treino)

print("Carregando classes")
identificacoes_treino = np.loadtxt("../freesound-audio-tagging/categorias.txt", delimiter='\n', dtype= 'int')

print("Inicializando modelo")
modelo = Sequential()
modelo.add(Conv2D(100, (2, 2), input_shape=(128, 8, 4), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Dropout(0,2))
modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(32, activation='relu'))
modelo.add(Dense(1, activation='softmax', name='predict'))
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

historico = modelo.fit(imagens_treino, identificacoes_treino, epochs=5, validation_split=0.2)