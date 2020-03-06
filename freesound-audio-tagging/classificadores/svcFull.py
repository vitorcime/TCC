import numpy as np
import pandas as pd

import scipy
import matplotlib.pyplot as plt
import urllib
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

def SVM(c):
    print("Carregando treino")
    train = np.loadtxt('trainArray.txt')
    print("Carregando teste")
    test = np.loadtxt('testArray.txt')
    print("Carregando categorias treino")
    categoriasTrain = np.loadtxt("categorias.txt", delimiter='\n', dtype= 'str')
    print("Carregando categorias teste")
    categoriasTest = np.loadtxt("categoriasFull.txt", delimiter='\n', dtype= 'str')

    print("normalizando...")
    ss = StandardScaler()
    train = ss.fit_transform(train)
    
    print("Treinando...")
    neigh = SVC(C=c, gamma='auto')
    neigh.fit(train, categoriasTrain)
    y_expect = categoriasTest
    test = ss.transform(test)
    print("Predict")
    y_pred = neigh.predict(test)

    
    print(metrics.classification_report(y_expect, y_pred))
    print("Metrica de c:" + str(c))

if __name__ == "__main__":
    SVM(100)

   