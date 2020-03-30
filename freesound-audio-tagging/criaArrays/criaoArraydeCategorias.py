import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   
import numpy as np
from multiprocessing import Pool

def CriaArray(img, lista, categoria):
    oi = glob.glob("../patchsTrain/"+  img.replace('.wav', '')+ "_*.png")
    for i in range(0, len(oi)):
        lista.append(categoria)
    return lista

def CriaArray2(args):
    img, categoria = args
    oi = glob.glob("../patchsTrain/"+  img.replace('.wav', '')+ "_*.png")
    return [categoria for i in range(len(oi))]

if __name__ == "__main__":
    arquivo = pd.read_csv('../CSV/audiosVerificados.csv')
    name = arquivo['fname']
    categorias = arquivo['label']
    lista = list()
    porcentagem = 0
    a = zip(name, categorias)
    b = sorted(a, key=lambda x: x[0])
    name, categorias = zip(*b)

    # for n in name[:5]:
    #     lista = CriaArray(n, lista, categorias[porcentagem])
    #     porcentagem+=1
    #     print(((porcentagem*100)/3710))
    #lista = array(lista)

    p = Pool(8)
    lista = p.map(CriaArray2, [(name[i], categorias[i]) for i in range(len(name))] )
    lista = np.concatenate(lista)

    print(lista.shape)
    np.savetxt('../categorias/categorias.txt', lista, newline='\n', fmt='%s')
    

    