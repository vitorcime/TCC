# -*- coding: utf-8 -*-

#from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Lambda, BatchNormalization, Activation 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import dill
import sys



    

scaler = StandardScaler()

# session_config=tf.compat.v1.ConfigProto(
#     gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7))
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_config))



print("Carregando imagens")
imagens_treino = np.load("../energiaSimples/trainArrayEnergia.npy")
#vamos tirar o canal alpha...
print(imagens_treino.shape)
imagens_treino = imagens_treino[:,:,:,:3]
imagens_treino = np.mean(imagens_treino, axis=-1, keepdims=True)
print(imagens_treino.shape)


print("Carregando classes")
identificacoes_treino = np.loadtxt("../../freesound-audio-tagging/categorias/categoriastrainEnergia.txt", delimiter='\n', dtype= 'str')
categorias = sorted(set(identificacoes_treino))
dic = dict()
for n, f in enumerate(categorias):
    dic[f] = n
for i in range(0, len(identificacoes_treino)):
    identificacoes_treino[i] = dic[identificacoes_treino[i]]
identificacoes_treino = to_categorical(identificacoes_treino)


X_train, X_val, Y_train, Y_val = train_test_split(imagens_treino, identificacoes_treino, test_size=0.2, 
    random_state=999, shuffle=True, stratify=np.argmax(identificacoes_treino, axis=1))

np.save('x_val.npy',X_val)
np.save('y_val.npy',Y_val)
print(X_train.shape)
train_shape = X_train.shape
X_train = np.reshape(X_train, (X_train.shape[0], -1))
val_shape= X_val.shape
X_val = np.reshape(X_val, (X_val.shape[0], -1))
print(X_train.shape)

ss = StandardScaler()
print("Fit")
ss.fit(X_train)
dill.dump(ss, open('ss.dill','wb'))

print("Transform")
X_train = ss.transform(X_train)
X_val = ss.transform(X_val)

print("Reshape")
X_train = np.reshape(X_train, train_shape)
X_val = np.reshape(X_val, val_shape)
print(tf.__version__)


print("Inicializando modelo")
'''
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
ipt = Input(shape=(64, int(sys.argv[2]), 1) )
l = BatchNormalization()(ipt)
l = Conv2D(100, (7, 7),padding='same', strides=1, activation='linear')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = MaxPooling2D(pool_size=(3, 3),strides=2, padding='same')(l)
l = Conv2D(150, (5, 5), activation='linear', strides=1, padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(l)
l = Conv2D(200, (3, 3), activation='linear', strides=1, padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Lambda(lambda x: K.max(x, axis=[1,2], keepdims=True), name='ReduceMax')(l)
l = Flatten()(l)
l = Dense(41, activation='softmax')(l)
model = keras.models.Model(inputs=ipt, outputs=l)
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

ES = EarlyStopping(monitor='val_loss', patience=40, verbose=1, min_delta=0.001, restore_best_weights=True)
CK = ModelCheckpoint("modeloSemDA", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
historico = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=64, callbacks=[ES,CK], epochs=1000)

plt.plot(historico.history['loss'], label='Training')
plt.plot(historico.history['val_loss'], label='Validation')
plt.legend()
plt.savefig("train_curve.png")
model.save("./modelosEnergiaSimples/modelo" + sys.argv[1] + ' ' + str(sys.argv[2]))

