import os
import numpy as np
import pandas as pd
from PIL import Image
from numpy import array
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import copy
import shutil
from multiprocessing import Pool
import sys
import glob

# TODO: colocar parametro n_frames e tipo_limiar (media, media - std, media - (std/2))
energia = sys.argv[1]
def calculaEnergiaTrain(img, lista,n_frames, tipo_limiar):
    img = img.replace('.wav', '.png')
    imagem = Image.open('../freesound-audio-tagging/specsTrain/'+img)
    patches = []
    arr = array(imagem)
    arr = arr[:,:,:3]
    arr = np.mean(arr, axis=-1, keepdims=True)
    soma = np.sum(np.square(arr), axis=0)
    soma[:] = [x / soma[np.argmax(soma)] for x in soma]

    #passar um filtro passa-baixas para suavizar os "picos" altos e baixos
    k = np.ones(10) / 10
    soma_c = np.convolve(soma[:,0], k)

    #calcular o limiar
    media_soma = np.mean(soma_c)
    std_soma = np.std(soma_c)
    limiar = float(energia)
    # if tipo_limiar == 0:
    #     limiar =  media_soma
    # elif tipo_limiar == 1:
    #     limiar = media_soma - std_soma
    # elif tipo_limiar == 2:
    #     limiar = media_soma - (std_soma / 2)
        

    #sel é um vetor de frames "selecionados". Nesse caso, o threshold é np.mean(soma_c). Os frames que excedem o treshold passam a ser 1, enquanto os demais
    #se tornam 0.
    sel = copy.copy(soma_c)
    sel[sel>=limiar] = 1
    sel[sel<1] = 0
    
    if 1 not in sel:
        soma_c = [x / soma_c[np.argmax(soma_c)] for x in soma_c]
        sel = copy.copy(soma_c)
        sel = np.array(sel)
        sel[sel>=limiar] = 1
        sel[sel<1] = 0
    
    patches = sorted(list(set(filter(None,[ int(i / (int(n_frames)/2)) if (sel[i] and i+int(n_frames)<soma.shape[0]) else None for i in range(soma.shape[0])]))))
    for i in patches:
        try:
            patch = Image.open('../freesound-audio-tagging/patchsTrain'+ str(n_frames) + '/' + img.replace('.png', '_') + str(i) + '.png' )
            arr =  array(patch) 
            lista.append(arr) 
        except:
            None
    return lista

def calculaEnergiaTest(img, lista,n_frames, tipo_limiar): 
    img = img.replace('.wav', '.png')
    imagem = Image.open('../freesound-audio-tagging/specsTest/'+img)
    patches = []
    arr = array(imagem)
    arr = arr[:,:,:3]
    arr = np.mean(arr, axis=-1, keepdims=True)
    soma = np.sum(np.square(arr), axis=0)
    soma[:] = [x / soma[np.argmax(soma)] for x in soma]

    #passar um filtro passa-baixas para suavizar os "picos" altos e baixos
    k = np.ones(10) / 10
    soma_c = np.convolve(soma[:,0], k)

    #calcular o limiar
    media_soma = np.mean(soma_c)
    std_soma = np.std(soma_c)
    limiar = float(energia)
    # if tipo_limiar == 0:
    #     limiar =  media_soma
    # elif tipo_limiar == 1:
    #     limiar = media_soma - std_soma
    # elif tipo_limiar == 2:
    #     limiar = media_soma - (std_soma / 2)
        

    #sel é um vetor de frames "selecionados". Nesse caso, o threshold é np.mean(soma_c). Os frames que excedem o treshold passam a ser 1, enquanto os demais
    #se tornam 0.
    sel = copy.copy(soma_c)
    sel[sel>=limiar] = 1
    sel[sel<1] = 0
    if 1 not in sel:
        soma_c = [x / soma_c[np.argmax(soma_c)] for x in soma_c]
        sel = copy.copy(soma_c)
        sel = np.array(sel)
        sel[sel>=limiar] = 1
        sel[sel<1] = 0
    
    patches = sorted(list(set(filter(None,[ int(i / (int(n_frames)/2)) if (sel[i] and i+int(n_frames)<soma.shape[0]) else None for i in range(soma.shape[0])]))))


    
    for i in patches:
        try:
            patch = Image.open('../freesound-audio-tagging/patchsTest'+ str(n_frames) + '/' +img.replace('.png', '_') + str(i) + '.png' )
            arr =  array(patch) 
            lista.append(arr) 
        except:
            None
    return lista


arquivo = pd.read_csv('../freesound-audio-tagging/CSV/audiosVerificados.csv')
name = sorted(arquivo['fname']) 
lista = list()
porcentagem = 0
for n in name:
    lista = calculaEnergiaTrain(n, lista, sys.argv[2], sys.argv[1])
    porcentagem+=1
    print( "%.3f" % ((porcentagem*100)/len(name)))
lista = array(lista)
print(lista.shape)
np.save("./trainArrayEnergia.npy", lista)

arquivo = pd.read_csv('../freesound-audio-tagging/CSV/test_post_competition.csv')
name = sorted(arquivo['fname']) 
lista = list()
porcentagem = 0
for n in name:
    lista = calculaEnergiaTest(n, lista, sys.argv[2], sys.argv[1])
    porcentagem+=1
    print( "%.3f" % ((porcentagem*100)/len(name)))
lista = array(lista)
print(lista.shape)
np.save("testArrayEnergia.npy", lista)



