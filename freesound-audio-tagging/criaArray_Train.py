import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   

def CriaArray(img, lista):
    oi = glob.glob("./patchsTrain/"+  img.replace('.wav', '')+ "_*.png")
    for i in oi:
        imagem = Image.open(i)
        arr =  array(imagem) 
        arr = arr.sum(axis=2)/4
        arr = arr.sum(axis=1)/8
        lista.append(arr) 
    return lista

if __name__ == "__main__":
    arquivo = pd.read_csv('train_post_competition.csv')
    name = arquivo['fname']
    lista = list()
    porcentagem = 0
    for n in name[:5]:
        lista = CriaArray(n, lista)
        porcentagem+=1
        print(((porcentagem*100)/5))
    lista = array(lista)
    np.savetxt('trainArray.txt', lista, newline='\n')
    

    