# -*- coding: utf-8 -*-
from keras.backend.tensorflow_backend import set_session
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
from keras.callbacks import EarlyStopping
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

session_config=tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_config))


print("Carregando imagens")
imagens_treino = np.load("lista.npy")
imagens_treino = np.mean(imagens_treino, axis=-1, keepdims=True)


print("Carregando classes")
identificacoes_treino = np.loadtxt("../freesound-audio-tagging/categorias.txt", delimiter='\n', dtype= 'str')
categorias = sorted(set(identificacoes_treino))
dic = dict()
for n, f in enumerate(categorias):
    dic[f] = n
for i in range(0, len(identificacoes_treino)):
    identificacoes_treino[i] = dic[identificacoes_treino[i]]
identificacoes_treino = to_categorical(identificacoes_treino)


print("Inicializando modelo")
modelo = Sequential()
modelo.add(Conv2D(100, (2, 2), input_shape=(128, 8, 1), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Conv2D(150, (2, 2), input_shape=(128, 8, 1), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Dropout(0.2))
modelo.add(Flatten())
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(41, activation='softmax', name='predict'))
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


ES = EarlyStopping(monitor='val_loss', patience=40, verbose=1, min_delta=0.001, restore_best_weights=True)
historico = modelo.fit(imagens_treino, identificacoes_treino,batch_size=1000, callbacks=[ES], epochs=1000, validation_split=0.2)
