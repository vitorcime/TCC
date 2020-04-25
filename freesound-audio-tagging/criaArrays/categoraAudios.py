import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   

def CriaArray(img, lista):
    quantidade = glob.glob("../patchsTrain/"+  img.replace('.wav', '')+ "_*.png")
    lista.append(len(quantidade))
    return lista

if __name__ == "__main__":
    arquivo = pd.read_csv('../audiosEscolhidos.csv')
    name = sorted(arquivo['fname'])
    lista = list()
    porcentagem = 0
    for n in name:
        lista = CriaArray(n, lista)
        porcentagem+=1
        print("%.3f" % ((porcentagem*100)/len(name)))
    lista = array(lista)
    print(lista.shape)
    np.savetxt('numeroDePatches.txt', lista, newline='\n', fmt='%s')
    


