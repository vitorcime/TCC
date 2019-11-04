import numpy as np

import scipy
import matplotlib.pyplot as plt
import urllib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

def KNN(k):
    rs = 0
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
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train, categoriasTrain)
    print("Fazendo o predict")
    test = ss.transform(test)
    y_pred = neigh.predict(test)

    print("Métricas")
    print(metrics.classification_report(categoriasTest, y_pred))
    print("Metrica de k:" + str(k))

if __name__ == "__main__":
    KNN(3)
   