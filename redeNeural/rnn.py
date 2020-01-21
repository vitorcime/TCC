# -*- coding: utf-8 -*-

import tensorflow
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image 
names = os.listdir('C:/Users/Pichau/Desktop/TCC/freesound-audio-tagging/patchsTrain')
imagens_treino = []

for i in names[0:46395]:
    arr = Image.open(r"../freesound-audio-tagging/patchsTrain/" + i)
    arr = np.array(arr)
    imagens_treino.append(arr)
imagens_treino = np.asarray(imagens_treino)
identificacoes_treino = np.loadtxt("../freesound-audio-tagging/categorias.txt", delimiter='\n', dtype= 'int')


modelo = keras.Sequential([
  keras.layers.Flatten(input_shape = (128, 8, 4)),
  keras.layers.Dense(256, activation=tensorflow.nn.relu),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(activation=tensorflow.nn.softmax)
])
modelo.compile(optimizer='adam', 
               loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

historico = modelo.fit(imagens_treino, identificacoes_treino, epochs=5, validation_split=0.2)