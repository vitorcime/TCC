import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   
import numpy as np
from multiprocessing import Pool
import sys 


def CriaArray(args):
    img, categoria = args
    if (sys.argv[1] == 'train'):
        oi = glob.glob("../patchsTrain50/"+  img.replace('.wav', '')+ "_*.png")
    if (sys.argv[1] == 'test'):
        oi = glob.glob("../patchsTest50/"+  img.replace('.wav', '')+ "_*.png")    
    return [categoria for i in range(len(oi))]

if __name__ == "__main__":
    if (sys.argv[1] == 'train'):
        arquivo = pd.read_csv('../CSV/audiosVerificados.csv')
    if (sys.argv[1] == 'test'):
        arquivo = pd.read_csv('../CSV/test_post_competition.csv')    
        
    name = arquivo['fname']
    categorias = arquivo['label']
    lista = list()
    a = zip(name, categorias)
    b = sorted(a, key=lambda x: x[0])
    print(len(b))
    name, categorias = zip(*b)
    p = Pool(8)
    print("Map")
    print(len(categorias))
    print(len(name))
    lista = p.map(CriaArray, [(name[i], categorias[i]) for i in range(len(name))] )
    lista = np.concatenate(lista)
    print(lista.shape)
    np.savetxt('../categorias/categorias'+ sys.argv[1] +'.txt', lista, newline='\n', fmt='%s')


