#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import glob
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import time
import numpy as np
import keras
from keras.layers import Input, Dense
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from PIL import Image
from numpy import array

# # Carregar Dados
# 
# Isso só faz sentido nesse exemplo. Provavelmente quando usarmos isso com muitos dados o que vamos passar como X é o nome dos arquivos.
# 

# In[2]:

print("Carregando imagens")
patchsNames = np.loadtxt("nomes.txt", delimiter='\n', dtype= 'str')


print("Carregando classes")
identificacoes_treino = np.loadtxt("../freesound-audio-tagging/categorias/categoriasTrain.txt", delimiter='\n', dtype= 'str')
categorias = sorted(set(identificacoes_treino))
dic = dict()
for n, f in enumerate(categorias):
    dic[f] = n
for i in range(0, len(identificacoes_treino)):
    identificacoes_treino[i] = dic[identificacoes_treino[i]]
identificacoes_treino = to_categorical(identificacoes_treino)

X_train, X_val, Y_train, Y_val = train_test_split(patchsNames, identificacoes_treino, test_size=0.2)


# # Definição da Classe DataGenerator

# In[12]:


class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, labels, batch_size=32, dim=(64,26,1), shuffle=True, mixup_alpha=0.3):
        self.X = X
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.indexes = None
        self.shuffle = shuffle
        self.mixup_alpha = mixup_alpha
        self.on_epoch_end()

    def __len__(self):
        #Serve pra indicar quantos batches terão por época. Quem chama isso é o keras.
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        #Cada thread chama essa função pra gerar os batches!
        
        #esse self.indexes contém os índices de todos os exemplos do dataset
        #já indexes contém uma subsequencia de self.indexes que corresponde ao batch atual
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        #chama a função que gera os dados. na vdd não precisaria chamar outra
        #função pra isso..
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        #isso é executado ao final de toda época (keras que chama)
        #Note que self.indexes é apenas num numpy array, com 
            #o índice de cada elemento do dataset. Opcionalmente,
            #os índices são embaralhados.
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def mixup(self, x1, x2, alpha):
        #função de mixup
        return (alpha * x1) + ((1-alpha) * x2)
            
    def __data_generation(self, sample_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        #como vamos usar o mixup, a quantidade de elementos na saída é metade
        #do número de elementos do batch
        output_len = int(np.floor(self.batch_size / 2))
        
        # pré-alocar os numpy arrays de saída.
        # Isso é mais rápido que ir fazendo append na lista e depois
        # chamando np.array pra transformar um um numpy array.
        X = np.empty((output_len, *self.dim))
        y = np.empty((output_len, *self.labels.shape[1:]))
        
        #pra cada elemento da saída
        for i in range(output_len):
            #computa o índice que contém o indice das 2 imagens
            j = sample_indexes[2*i]
            k = sample_indexes[(2*i)+1]
            
            #calcula o mixup
            #Note que em um caso mais real, não teríamos o X... o X seria apenas um vetor com o nome
            #dos arquivos que contém as imagens! Daí teria que abrir os arquivos, extrair as matrizes
            #de pixels, e daí sim, chamar o mixup.
            img1 = Image.open(self.X[j])
            img1 =  array(img1)
            img1 = img1[:,:,:3]
            img1 = np.mean(img1, axis=-1, keepdims=True)
            img2 = Image.open(self.X[k])
            img2 =  array(img2)
            img2 = img2[:,:,:3]
            img2 = np.mean(img2, axis=-1, keepdims=True) 
            X[i,] = self.mixup(img1, img2, self.mixup_alpha)
            y[i,] = self.mixup(self.labels[j], self.labels[k], self.mixup_alpha)
        
        #retorna o batch
        return X, y


# # Treinamento com Generator

# In[15]:


keras.backend.clear_session()

ipt = Input(shape=(64, 26, 1) )
net = Dense(100, activation='relu')(ipt)
net = Dense(100, activation='relu')(net)
opt = Dense(10, activation='softmax')(net)

model = keras.models.Model(inputs=ipt, outputs=opt)
model.compile(optimizer=keras.optimizers.Adam(lr=0.005), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

#Instancia os 2 generators, um para os dados de treino e outro para os dados de validação.
#Atenção ao parâmetro dim.. Ele indica qual é o shape de cada imagem e deve ser passado.
training_generator = DataGenerator(X_train, Y_train, batch_size=32, shuffle=True, mixup_alpha=0.1)
validation_generator = DataGenerator(X_val, Y_val, batch_size=32, shuffle=True, mixup_alpha=0.1)

t0 = time.time()
#Note que agora o fit recebe os generators ao invés dos dados diretamente.
#use_multiprocessing permite calcular os batches em paralelo. Entretanto, só é útil pra quando
#temos muitas imagens pra carregar e fazer mixup. No caso desse exemplo ele deixa o código mais
#demorado rs.
h = model.fit(training_generator, epochs=10, validation_data=validation_generator,
                       use_multiprocessing=False, workers=4, verbose=0)
print("O treino demorou %.2f segundos." % (time.time() - t0))



# plt.plot(h.history['loss'], label='Training loss')
# plt.plot(h.history['val_loss'], label='Validation loss')
# plt.legend()

