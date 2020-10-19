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
from sklearn.metrics import classification_report
import dill
scaler = StandardScaler()

session_config=tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_config))

print("Carregando imagens")
imagens_teste = np.load("../testArrayEnergia.npy")

#vamos tirar o canal alpha...
imagens_teste = imagens_teste[:,:,:,:3]
imagens_teste = np.mean(imagens_teste, axis=-1, keepdims=True)
print(imagens_teste.shape)



test_shape = imagens_teste.shape
imagens_teste = np.reshape(imagens_teste, (imagens_teste.shape[0], -1))



print("Transform")
ss = dill.load(open('ss.dill','rb'))
imagens_teste = ss.transform(imagens_teste)


print("Reshape")
imagens_teste = np.reshape(imagens_teste, test_shape)



modelo = load_model("../modelosEnergiaQuantificada/modelo0.9")
resultado = modelo.predict(imagens_teste)
y = []
for i in resultado:
   y.append(np.argmax(i))
print(np.unique(y, return_counts=True))





numerodePatches = np.loadtxt("../numeroPatchesEnergia.txt")

i = 0
resultadoSoma = list()
'''
for y in numerodePatches:
    resultadoSoma.append(np.sum(resultado[i:i+int(y),:], axis=0))
    i+=int(y)
resultadoSoma = np.array(resultadoSoma)
resultadoSoma = np.argmax(resultadoSoma, axis=1)
print(resultadoSoma.shape)
'''
print("Carregando classes")
identificacoes_teste = np.loadtxt("../../freesound-audio-tagging/categorias/categoriastestEnergia.txt", delimiter='\n', dtype= 'str')
resultado = np.argmax(resultado, axis=1)


categorias = sorted(set(identificacoes_teste))
dic = dict()
for n, f in enumerate(categorias):
    dic[f] = n
for i in range(0, len(identificacoes_teste)):
    identificacoes_teste[i] = dic[identificacoes_teste[i]]

identificacoes_teste = to_categorical(identificacoes_teste)
print(identificacoes_teste.shape)

identificacoes_teste = np.argmax(identificacoes_teste, axis=1)
print(identificacoes_teste.shape)
print(resultado.shape)
print(classification_report(y_pred = resultado, y_true = identificacoes_teste))

'''
perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)
print('Perda do teste: ', perda_teste)
print('Acuracia do teste: ', acuracia_teste)
'''