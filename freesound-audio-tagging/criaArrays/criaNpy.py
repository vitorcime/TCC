import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array
import sys

def CriaArray(img, lista, tipoDado):
    patches = glob.glob("../patchs" + tipoDado + "/"+  img.replace('.wav', '')+ "_*.png")
    for i in patches:
        imagem = Image.open(i)
        arr =  array(imagem) 
        lista.append(arr) 
    return lista

if (sys.argv[1] == 'train'):
    arquivo = pd.read_csv('../CSV/audiosVerificados.csv')
    name = sorted(arquivo['fname']) 
    lista = list()
    porcentagem = 0
    for n in name:
        lista = CriaArray(n, lista, 'Train')
        porcentagem+=1
        print( "%.3f" % ((porcentagem*100)/len(name)))
    lista = array(lista)
    print(lista.shape)
    np.save("../../redeNeural/trainArrayVerificados.npy", lista)
    
if (sys.argv[1] == 'test'):
    arquivo = pd.read_csv('../CSV/test_post_competition.csv')
    name = sorted(arquivo['fname']) 
    lista = list()
    porcentagem = 0
    for n in name:
        lista = CriaArray(n, lista, 'Test')
        porcentagem+=1
        print( "%.3f" % ((porcentagem*100)/len(name)))
    lista = array(lista)
    print(lista.shape)
    np.save("../../redeNeural/testArray.npy", lista)
    

    