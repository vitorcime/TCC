import glob
import os
import sys 
import numpy as np
import numpy as np
import pandas as pd
from PIL import Image
from numpy import array   
from itertools import groupby
from multiprocessing import Pool

energia = 0.9

def CriaArrayEnergiaTrain(args):
    img, categoria = args
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
    chosen = []
    for i in patches:
        if(len(i) >= 3):
            chosen.append(i[0])    
    return [categoria for i in range(len(chosen))]

def CriaArrayEnergiaTest(args):
    img, categoria = args
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
    chosen = []
    for i in patches:
        if(len(i) >= 3):
            chosen.append(i[0])    
    return [categoria for i in range(len(chosen))]

if __name__ == "__main__": 
    arquivo = pd.read_csv('../../freesound-audio-tagging/CSV/audiosVerificados.csv')  
    name = arquivo['fname']
    categorias = arquivo['label']
    lista = list()
    a = zip(name, categorias)
    b = sorted(a, key=lambda x: x[0])
    name, categorias = zip(*b)
    p = Pool(8)
    print("Map")
    lista = p.map(CriaArrayEnergiaTrain, [(name[i], categorias[i]) for i in range(len(name))] )
    lista = np.concatenate(lista)
    print(lista.shape)
    np.savetxt('../../freesound-audio-tagging/categorias/categoriastrainEnergia.txt', lista, newline='\n', fmt='%s')

if __name__ == "__main__": 
    arquivo = pd.read_csv('../../freesound-audio-tagging/CSV/test_post_competition.csv')  
    name = arquivo['fname']
    categorias = arquivo['label']
    lista = list()
    a = zip(name, categorias)
    b = sorted(a, key=lambda x: x[0])
    name, categorias = zip(*b)
    p = Pool(8)
    print("Map")
    lista = p.map(CriaArrayEnergiaTest, [(name[i], categorias[i]) for i in range(len(name))] )
    lista = np.concatenate(lista)
    print(lista.shape)
    np.savetxt('../../freesound-audio-tagging/categorias/categoriastestEnergia.txt', lista, newline='\n', fmt='%s')


