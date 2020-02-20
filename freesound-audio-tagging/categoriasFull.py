import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   

def CriaArray(img, lista, categoria):
    quantidade = glob.glob("./patchsTrain/"+  img.replace('.wav', '')+ "_*.png")
    for i in range(0, len(quantidade)):
        lista.append(categoria)
    return lista

if __name__ == "__main__":
    arquivo = pd.read_csv('train_post_competition.csv')
    name = arquivo['fname']
    categorias = arquivo['label']
    lista = list()
    porcentagem = 0
    for n in name:
        lista = CriaArray(n, lista, categorias[porcentagem])
        porcentagem+=1
        print(((porcentagem*100)/1600))
    lista = array(lista)
    print(lista.shape)
    np.savetxt('categoriasFull.txt', lista, newline='\n', fmt='%s')
    

    