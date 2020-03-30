# -*- coding: utf-8 -*-
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
import tensorflow as tf
import keras
import numpy as np
import os
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Lambda 
from keras.callbacks import EarlyStopping
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.backend as K

scaler = StandardScaler()

session_config=tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_config))


print("Carregando imagens")
imagens_treino = np.load("trainArrayVerificados.npy")
#vamos tirar o canal alpha...
imagens_treino = imagens_treino[:,:,:,:3]
imagens_treino = np.mean(imagens_treino, axis=-1, keepdims=True)
print(imagens_treino.shape)
train_shape = imagens_treino.shape
imagens_treino = np.reshape(imagens_treino, (imagens_treino.shape[0], -1))
print(imagens_treino.shape)
ss = StandardScaler()
print("Fit")
ss.fit(imagens_treino)
print("Transform")
imagens_treino = ss.transform(imagens_treino)
print("Reshape")
imagens_treino = np.reshape(imagens_treino, train_shape)

print("Carregando classes")
identificacoes_treino = np.loadtxt("../freesound-audio-tagging/categorias/categorias.txt", delimiter='\n', dtype= 'str')
categorias = sorted(set(identificacoes_treino))
dic = dict()
for n, f in enumerate(categorias):
    dic[f] = n
for i in range(0, len(identificacoes_treino)):
    identificacoes_treino[i] = dic[identificacoes_treino[i]]
identificacoes_treino = to_categorical(identificacoes_treino)


print("Inicializando modelo")
ipt = Input(shape=(64, 26, 1) )
l = Conv2D(100, (7, 7),padding='same', strides=1, activation='relu')(ipt)
l = MaxPooling2D(pool_size=(3, 3),strides=2, padding='same')(l)
l = Conv2D(150, (5, 5), activation='relu', strides=1, padding='same')(l)
l = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(l)
l = Conv2D(200, (3, 3), activation='relu', strides=1, padding='same')(l)
l = Lambda(lambda x: K.max(x, axis=[1,2], keepdims=True), name='ReduceMax')(l)
l = Flatten()(l)
l = Dense(41, activation='softmax')(l)
modelo = keras.Model(inputs=ipt, outputs=l)
print(modelo.summary())

'''
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
'''
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, X_val, Y_train, Y_val = train_test_split(imagens_treino, identificacoes_treino, test_size=0.2, random_state=999, shuffle=True, stratify=np.argmax(identificacoes_treino, axis=1))

ES = EarlyStopping(monitor='val_loss', patience=40, verbose=30, min_delta=0.001, restore_best_weights=True)
historico = modelo.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=64, callbacks=[ES], epochs=1000, validation_split=0.2)

plt.plot(historico.history['loss'], label='Training')
plt.plot(historico.history['val_loss'], label='Validation')
plt.legend()
plt.savefig("train_curve.png")
#plt.show()
