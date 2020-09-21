#!/usr/bin/env python
# coding: utf-8

# In[1]:
import random
import copy
import tensorflow.keras.backend as K
from numpy import array
from PIL import Image
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda, BatchNormalization, Activation
import keras
import numpy as np
import time
import tensorflow as tf
import glob
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.get_logger().setLevel("ERROR")

'''
session_config=tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_config))
'''
print("Carregando imagens")
patchsNames = np.loadtxt("nomes.txt", delimiter='\n', dtype='str')


print("Carregando classes")
identificacoes_treino = np.loadtxt(
    "../freesound-audio-tagging/categorias/categoriastrain.txt", delimiter='\n', dtype='str')
categorias = sorted(set(identificacoes_treino))
dic = dict()
for n, f in enumerate(categorias):
    dic[f] = n
for i in range(0, len(identificacoes_treino)):
    identificacoes_treino[i] = dic[identificacoes_treino[i]]

rotulos_treino = copy.deepcopy(identificacoes_treino)

identificacoes_treino = to_categorical(identificacoes_treino)

# é bom tentar estratificar: manter a proporção de elementos de cada classe a mesma no treino e na validação.
# pra isso usamos o parametro stratify, passando as labels de cada entrada (mas tem que ser no formato do sklearn)
X_train, X_val, Y_train, Y_val = train_test_split(
    patchsNames, identificacoes_treino, test_size=0.2, stratify=rotulos_treino)

print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)


# # Definição da Classe DataGenerator

# In[12]:


class MixupDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, labels, batch_size=32, dim=(64, 26, 1), shuffle=True, mixup_alpha=0.3):
        self.X = X
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.indexes = None
        self.shuffle = shuffle
        self.mixup_alpha = mixup_alpha
        self.on_epoch_end()

    def __len__(self):
        # Serve pra indicar quantos batches terão por época. Quem chama isso é o keras.
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # Cada thread chama essa função pra gerar os batches!

        # esse self.indexes contém os índices de todos os exemplos do dataset
        # já indexes contém uma subsequencia de self.indexes que corresponde ao batch atual
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # chama a função que gera os dados. na vdd não precisaria chamar outra
        # função pra isso..
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        # isso é executado ao final de toda época (keras que chama)
        # Note que self.indexes é apenas num numpy array, com
        # o índice de cada elemento do dataset. Opcionalmente,
        # os índices são embaralhados.
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def mixup(self, x1, x2, alpha):
        # função de mixup

        lbd = np.random.beta(alpha, alpha)

        return (lbd * x1) + ((1-lbd) * x2)

    def __data_generation(self, sample_indexes):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # como vamos usar o mixup, a quantidade de elementos na saída é metade
        # do número de elementos do batch
        output_len = int(np.floor(self.batch_size / 2))

        # pré-alocar os numpy arrays de saída.
        # Isso é mais rápido que ir fazendo append na lista e depois
        # chamando np.array pra transformar um um numpy array.
        # X = np.empty((output_len * 3, *self.dim))
        # y = np.empty((output_len * 3, *self.labels.shape[1:]))

        X = np.empty((output_len, *self.dim))
        y = np.empty((output_len, *self.labels.shape[1:]))

        #print("".join([" " for i in range(random.randint(0,10))])  + "mixing up...")

        # pra cada elemento da saída
        for i in range(output_len):
            # computa o índice que contém o indice das 2 imagens
            j = sample_indexes[2*i]
            k = sample_indexes[(2*i)+1]

            # calcula o mixup
            # Note que em um caso mais real, não teríamos o X... o X seria apenas um vetor com o nome
            # dos arquivos que contém as imagens! Daí teria que abrir os arquivos, extrair as matrizes
            # de pixels, e daí sim, chamar o mixup.
            img1 = Image.open(self.X[j].replace("\\", os.path.sep))
            img1 = array(img1)
            img1 = img1[:, :, :3]
            img1 = np.mean(img1, axis=-1, keepdims=True)
            img2 = Image.open(self.X[k].replace("\\", os.path.sep))
            img2 = array(img2)
            img2 = img2[:, :, :3]
            img2 = np.mean(img2, axis=-1, keepdims=True)
            X[i, ] = self.mixup(img1, img2, self.mixup_alpha)
            y[i, ] = self.mixup(
                self.labels[j], self.labels[k], self.mixup_alpha)

            # incluir as imagens originais também
            #X[ (1*output_len) + i,] = img1
            #X[ (2*output_len) + i,] = img2
            #y[ (1*output_len) + i,] = self.labels[j]
            #y[ (2*output_len) + i,] = self.labels[k]

        # retorna o batchcheckpointRnn005
        return X, y


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, X, labels, batch_size=32, dim=(64, 26, 1), shuffle=True):
        self.X = X
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.indexes = None
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # Cada thread chama essa função pra gerar os batches!

        # esse self.indexes contém os índices de todos os exemplos do dataset
        # já indexes contém uma subsequencia de self.indexes que corresponde ao batch atual
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # chama a função que gera os dados. na vdd não precisaria chamar outra
        # função pra isso..
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        # isso é executado ao final de toda época (keras que chama)
        # Note que self.indexes é apenas num numpy array, com
        # o índice de cada elemento do dataset. Opcionalmente,
        # os índices são embaralhados.
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sample_indexes):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # como vamos usar o mixup, a quantidade de elementos na saída é metade
        # do número de elementos do batch
        output_len = self.batch_size

        # pré-alocar os numpy arrays de saída.
        # Isso é mais rápido que ir fazendo append na lista e depois
        # chamando np.array pra transformar um um numpy array.
        X = np.empty((output_len, *self.dim))
        y = np.empty((output_len, *self.labels.shape[1:]))

        #print("".join([" " for i in range(random.randint(0,10))])  + "mixing up...")

        # pra cada elemento da saída
        for i in range(output_len):
            # computa o índice que contém o indice das 2 imagens
            j = sample_indexes[i]

            # calcula o mixup
            # Note que em um caso mais real, não teríamos o X... o X seria apenas um vetor com o nome
            # dos arquivos que contém as imagens! Daí teria que abrir os arquivos, extrair as matrizes
            # de pixels, e daí sim, chamar o mixup.
            img1 = Image.open(self.X[j].replace("\\", os.path.sep))
            img1 = array(img1)
            img1 = img1[:, :, :3]
            img1 = np.mean(img1, axis=-1, keepdims=True)

            X[i, ] = img1
            y[i, ] = self.labels[j]

        # retorna o batch
        return X, y


# # Treinamento com Generator

# In[15]:


tf.keras.backend.clear_session()
''''
ipt = Input(shape=(64, 26, 1) )
net = Dense(100, activation='relu')(ipt)
net = Dense(100, activation='relu')(net)
opt = Dense(10, activation='softmax')(net)
'''
print("Inicializando modelo")
ipt = Input(shape=(64, 26, 1))
l = BatchNormalization()(ipt)
l = Conv2D(100, (7, 7), padding='same', strides=1, activation='linear')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(l)
l = Conv2D(150, (5, 5), activation='linear', strides=1, padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(l)
l = Conv2D(200, (3, 3), activation='linear', strides=1, padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Lambda(lambda x: K.max(x, axis=[1, 2], keepdims=True), name='ReduceMax')(l)
l = Flatten()(l)
l = Dense(41, activation='softmax')(l)
model = tf.keras.models.Model(inputs=ipt, outputs=l)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Instancia os 2 generators, um para os dados de treino e outro para os dados de validação.
# Atenção ao parâmetro dim.. Ele indica qual é o shape de cada imagem e deve ser passado.
alpha = 0.05
training_generator = MixupDataGenerator(
    X_train, Y_train, batch_size=256, shuffle=True, mixup_alpha=alpha)
validation_generator = DataLoader(X_val, Y_val, batch_size=256, shuffle=True)

t0 = time.time()
# Note que agora o fit recebe os generators ao invés dos dados diretamente.
# use_multiprocessing permite calcular os batches em paralelo. Entretanto, só é útil pra quando
# temos muitas imagens pra carregar e fazer mixup. No caso desse exemplo ele deixa o código mais
# demorado rs.

ES = EarlyStopping(monitor='val_loss', patience=40, verbose=1,
                   min_delta=0.001, restore_best_weights=True)
CK = ModelCheckpoint("checkpointRnn005", monitor='val_loss',
                     verbose=1, save_best_only=True, mode='min')
h = model.fit(training_generator, epochs=500, validation_data=validation_generator,
              use_multiprocessing=True, workers=1, verbose=2, callbacks=[ES, CK])
print("O treino demorou %.2f segundos." % (time.time() - t0))
model.save("modeloMixup005")


# plt.plot(h.history['loss'], label='Training loss')
# plt.plot(h.history['val_loss'], label='Validation loss')
# plt.legend()
