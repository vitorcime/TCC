import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   
import numpy as np
from multiprocessing import Pool
import sys 
import glob
import copy

energia = sys.argv[1]
def calculaEnergiaTrain(args):
    img, categoria, n_frames, tipo_limiar = args 
    img = img.replace('.wav', '.png')
    imagem = Image.open('../freesound-audio-tagging/specsTrain/'+img)
    patches = []
    arr = array(imagem)
    arr = arr[:,:,:3]
    arr = np.mean(arr, axis=-1, keepdims=True)
    soma = np.sum(np.square(arr), axis=0)

    soma[:] = [x / soma[np.argmax(soma)] for x in soma]


    #passar um filtro passa-baixas para suavizar os "picos" altos e baixos
    k = np.ones(10) / 10
    soma_c = np.convolve(soma[:,0], k)

    #calcular o limiar
    media_soma = np.mean(soma_c)
    std_soma = np.std(soma_c)
    limiar = float(energia)
    # if tipo_limiar == 0:
    #     limiar =  media_soma
    # elif tipo_limiar == 1:
    #     limiar = media_soma - std_soma
    # elif tipo_limiar == 2:
    #     limiar = media_soma - (std_soma / 2)
        

    #sel é um vetor de frames "selecionados". Nesse caso, o threshold é np.mean(soma_c). Os frames que excedem o treshold passam a ser 1, enquanto os demais
    #se tornam 0.
    sel = copy.copy(soma_c)
    sel[sel>=limiar] = 1
    sel[sel<1] = 0
    if 1 not in sel:
        soma_c = [x / soma_c[np.argmax(soma_c)] for x in soma_c]
        sel = copy.copy(soma_c)
        sel = np.array(sel)
        sel[sel>=limiar] = 1
        sel[sel<1] = 0
    
    patches = sorted(list(set(filter(None,[ int(i / (int(n_frames)/2)) if (sel[i] and i+int(n_frames)<soma.shape[0]) else None for i in range(soma.shape[0])]))))
    lista = []
    for i in patches:
        try:
            patch = Image.open('../freesound-audio-tagging/patchsTrain'+ str(sys.argv[2])+'/'+img.replace('.png', '_') + str(i) + '.png' )
            arr =  array(patch) 
            lista.append(arr) 
        except:
            None
    return [categoria for i in range(len(lista))]
def calculaEnergiaTest(args):
    img, categoria, n_frames, tipo_limiar = args 
    img = img.replace('.wav', '.png')
    imagem = Image.open('../freesound-audio-tagging/specsTest/'+img)
    patches = []
    arr = array(imagem)
    arr = arr[:,:,:3]
    arr = np.mean(arr, axis=-1, keepdims=True)
    soma = np.sum(np.square(arr), axis=0)
    soma[:] = [x / soma[np.argmax(soma)] for x in soma]
    

    #passar um filtro passa-baixas para suavizar os "picos" altos e baixos
    k = np.ones(10) / 10
    soma_c = np.convolve(soma[:,0], k)

    #calcular o limiar
    media_soma = np.mean(soma_c)
    std_soma = np.std(soma_c)
    limiar = float(energia)
    # if tipo_limiar == 0:
    #     limiar =  media_soma
    # elif tipo_limiar == 1:
    #     limiar = media_soma - std_soma
    # elif tipo_limiar == 2:
    #     limiar = media_soma - (std_soma / 2)
        

    #sel é um vetor de frames "selecionados". Nesse caso, o threshold é np.mean(soma_c). Os frames que excedem o treshold passam a ser 1, enquanto os demais
    #se tornam 0.
    sel = copy.copy(soma_c)
    sel[sel>=limiar] = 1
    sel[sel<1] = 0
    if 1 not in sel:
        soma_c = [x / soma_c[np.argmax(soma_c)] for x in soma_c]
        sel = copy.copy(soma_c)
        sel = np.array(sel)
        sel[sel>=limiar] = 1
        sel[sel<1] = 0
    
    patches = sorted(list(set(filter(None,[ int(i / (int(n_frames)/2)) if (sel[i] and i+int(n_frames)<soma.shape[0]) else None for i in range(soma.shape[0])]))))
    lista = []
    for i in patches:
        try:
            patch = Image.open('../freesound-audio-tagging/patchsTest'+str(sys.argv[2])+'/'+img.replace('.png', '_') + str(i) + '.png' )
            arr =  array(patch) 
            lista.append(arr) 
        except:
            None
    return [categoria for i in range(len(lista))]

if __name__ == "__main__": 
    arquivo = pd.read_csv('../freesound-audio-tagging/CSV/audiosVerificados.csv')  
    name = arquivo['fname']
    categorias = arquivo['label']
    lista = list()
    a = zip(name, categorias)
    b = sorted(a, key=lambda x: x[0])
    name, categorias = zip(*b)
    p = Pool(8)
    print("Map")
    lista = p.map(calculaEnergiaTrain, [(name[i], categorias[i], str(sys.argv[2]), sys.argv[1]) for i in range(len(name))] )
    lista = np.concatenate(lista)
    print(lista.shape)
    np.savetxt('../freesound-audio-tagging/categorias/categoriastrainEnergia.txt', lista, newline='\n', fmt='%s')

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
    lista = p.map(calculaEnergiaTest, [(name[i], categorias[i], str(sys.argv[2]), sys.argv[1]) for i in range(len(name))] )
    lista = np.concatenate(lista)
    print(lista.shape)
    np.savetxt('../freesound-audio-tagging/categorias/categoriastestEnergia.txt', lista, newline='\n', fmt='%s')


