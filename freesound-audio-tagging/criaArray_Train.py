import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   

def CriaArray(img, lista):
    patches = glob.glob("./patchsTrain/"+  img.replace('.wav', '')+ "_*.png")
    for i in patches:
        imagem = Image.open(i)
        arr =  array(imagem) 
        lista.append(arr) 
    return lista

if __name__ == "__main__":
    arquivo = pd.read_csv('audiosEscolhidos.csv')
    name = arquivo['fname']
    lista = list()
    porcentagem = 0
    for n in name:
        lista = CriaArray(n, lista)
        porcentagem+=1
        print(((porcentagem*100)/936))
    lista = array(lista)
    print(lista.shape)
    np.savetxt('trainArray.txt', lista, newline='\n')
    

    