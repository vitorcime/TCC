import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   

def CriaArray(img, lista):
    oi = glob.glob("./patchsTest/"+  img.replace('.wav', '')+ "_*.png")
    for i in oi:
        imagem = Image.open(i)
        arr =  array(imagem) 
        arr = arr.sum(axis=2)
        arr = arr.sum(axis=1)
        arr = arr/12
        lista.append(arr) 
    return lista

if __name__ == "__main__":
    arquivo = pd.read_csv('test_post_competition.csv')
    name = arquivo['fname']
    lista = list()
    porcentagem = 0
    for n in name:
        lista = CriaArray(n, lista)
        porcentagem+=1
        print(((porcentagem*100)/1600))
    lista = array(lista)
    np.savetxt('testArray.txt', lista, newline='\n')
    

    