from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Lambda 
from keras.callbacks import EarlyStopping
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.backend as K
from tensorflow.keras.models import load_model


scaler = StandardScaler()

session_config=tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_config))

print("Carregando imagens")
imagens_teste = np.load("testArray.npy")
#vamos tirar o canal alpha...
imagens_teste = imagens_teste[:,:,:,:3]
imagens_teste = np.mean(imagens_teste, axis=-1, keepdims=True)

print("Carregando classes")
identificacoes_teste = np.loadtxt("../freesound-audio-tagging/categorias/categoriastest.txt", delimiter='\n', dtype= 'str')
categorias = sorted(set(identificacoes_teste))
dic = dict()
for n, f in enumerate(categorias):
    dic[f] = n
for i in range(0, len(identificacoes_teste)):
    identificacoes_teste[i] = dic[identificacoes_teste[i]]
identificacoes_teste = to_categorical(identificacoes_teste)


test_shape = imagens_teste.shape
imagens_teste = np.reshape(imagens_teste, (imagens_teste.shape[0], -1))


ss = StandardScaler()
print("Transform")
ss.fit(imagens_teste)
imagens_teste = ss.transform(imagens_teste)


print("Reshape")
imagens_teste = np.reshape(imagens_teste, test_shape)



modelo = load_model("modelo")
resultado = modelo.predict(imagens_teste)
print(resultado.shape)


numerodePatches = np.loadtxt("numeroDePatches.txt")

i = 0
resultadoSoma = list()

for y in numerodePatches:
    print(i)
    print((resultado[i:int(y)]).shape)
    resultadoSoma.append(np.sum(resultado[i:int(y)], axis=0))
    i+=1
resultadoSoma = np.array(resultadoSoma)
print(resultadoSoma.shape)



'''
perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)
print('Perda do teste: ', perda_teste)
print('Acuracia do teste: ', acuracia_teste)
'''