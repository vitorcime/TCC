import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   
import numpy as np
from multiprocessing import Pool
import sys 


def CriaArrayEnergia(args):
    img, categoria = args
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
    return [categoria for i in range(len(patches))]

if __name__ == "__main__": 
    arquivo = pd.read_csv('../freesound-audio-tagging/CSV/test_post_competition.csv')  
    name = arquivo['fname']
    categorias = arquivo['label']
    lista = list()
    a = zip(name, categorias)
    b = sorted(a, key=lambda x: x[0])
    name, categorias = zip(*b)
    p = Pool(8)
    print("Map")
    lista = p.map(CriaArrayEnergia, [(name[i], categorias[i]) for i in range(len(name))] )
    lista = np.concatenate(lista)
    print(lista.shape)
    np.savetxt('../freesound-audio-tagging/categorias/categoriastestEnergia.txt', lista, newline='\n', fmt='%s')


