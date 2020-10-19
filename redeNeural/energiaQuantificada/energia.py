import os
import numpy as np
import pandas as pd
from PIL import Image
from numpy import array
from itertools import groupby
from joblib import Parallel, delayed

energia = 0.9

def calculaEnergiaTrain(img, lista):
    img = img.replace('.wav', '.png')
    imagem = Image.open('../../freesound-audio-tagging/specsTrain/'+img)
    patches = []
    arr = array(imagem)
    arr = arr[:,:,:3]
    arr = np.mean(arr, axis=-1, keepdims=True)
    soma = np.sum(np.square(arr), axis=0)
    soma[:] = [x / soma[np.argmax(soma)] for x in soma]
    for i in range(0,soma.shape[0]):
        if (soma[i] > energia and i+13 <= soma.shape[0]):
            patches.append(int(i/26))
    patches.sort()
    patches = (list(j) for i, j in groupby(patches))
    for i in patches:
        if(len(i) >= 3):
            i = i[0]
            patch = Image.open('../../freesound-audio-tagging/patchsTrain/'+img.replace('.png', '_') + str(i) + '.png' )
            arr =  array(patch) 
            lista.append(arr) 
    return lista

def calculaEnergiaTest(img, lista):
    img = img.replace('.wav', '.png')
    imagem = Image.open('../../freesound-audio-tagging/specsTest/'+img)
    patches = []
    arr = array(imagem)
    arr = arr[:,:,:3]
    arr = np.mean(arr, axis=-1, keepdims=True)
    soma = np.sum(np.square(arr), axis=0)
    soma[:] = [x / soma[np.argmax(soma)] for x in soma]
    for i in range(0,soma.shape[0]):
        if (soma[i] > energia and i+13 <= soma.shape[0]):
            patches.append(int(i/26))
    patches.sort()
    patches = (list(j) for i, j in groupby(patches))
    for i in patches:
        if(len(i) >= 3):
            i = i[0]
            patch = Image.open('../../freesound-audio-tagging/patchsTest/'+img.replace('.png', '_') + str(i) + '.png' )
            arr =  array(patch) 
            lista.append(arr) 
    return lista

arquivo = pd.read_csv('../../freesound-audio-tagging/CSV/audiosVerificados.csv')
name = sorted(arquivo['fname']) 
lista = list()
porcentagem = 0
for n in name:
    lista = calculaEnergiaTrain(n, lista)
    porcentagem+=1
    print( "%.3f" % ((porcentagem*100)/len(name)))
lista = array(lista)
print(lista.shape)
np.save("../trainArrayEnergia.npy", lista)

arquivo = pd.read_csv('../../freesound-audio-tagging/CSV/test_post_competition.csv')
name = sorted(arquivo['fname']) 
lista = list()
porcentagem = 0
for n in name:
    lista = calculaEnergiaTest(n, lista)
    porcentagem+=1
    print( "%.3f" % ((porcentagem*100)/len(name)))
lista = array(lista)
print(lista.shape)
np.save("../testArrayEnergia.npy", lista)