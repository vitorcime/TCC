import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   

def calculaEnergia(img, lista):
    img = img.replace('.wav', '.png')
    imagem = Image.open('../freesound-audio-tagging/specsTest/'+img)
    patches = []
    arr = array(imagem)
    arr = arr[:,:,:3]
    arr = np.mean(arr, axis=-1, keepdims=True)
    soma = np.sum(np.square(arr), axis=0)
    soma[:] = [x / soma[np.argmax(soma)] for x in soma]
    for i in range(0,soma.shape[0]):
        if (soma[i] > 0.3 and i+13 <= soma.shape[0]):
            patches.append(int(i/26))
    patches = set(patches)
    lista.append(patches)
    return lista

if __name__ == "__main__":
    arquivo = pd.read_csv('../freesound-audio-tagging/CSV/test_post_competition.csv')
    name = sorted(arquivo['fname'])
    lista = list()
    porcentagem = 0
    for n in name:
        lista = calculaEnergia(n, lista)
        porcentagem+=1
        print("%.3f" % ((porcentagem*100)/len(name)))
    lista = array(lista)
    print(lista.shape)
    numeros = []
    for i in lista:
        numeros.append(len(i))
    np.savetxt('patchesEnergia.txt', lista, newline='\n', fmt='%s')
    np.savetxt('numeroPatchesEnergia.txt', numeros, newline='\n', fmt='%s')


