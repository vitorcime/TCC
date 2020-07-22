import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   

def CriaArray(img, lista):
    nomes = glob.glob("../freesound-audio-tagging/patchsTrain/"+  img.replace('.wav', '')+ "_*.png")
    print(nomes)
    lista.extend(nomes)
    return lista

if __name__ == "__main__":
    arquivo = pd.read_csv('../freesound-audio-tagging/audiosEscolhidos.csv')
    name = sorted(arquivo['fname'])
    lista = list()
    porcentagem = 0
    for n in name:
        lista = CriaArray(n, lista)
        porcentagem+=1
        print("%.3f" % ((porcentagem*100)/len(name)))
    lista = array(lista)
    np.savetxt('nomes.txt', lista, newline='\n', fmt='%s')
    
