import numpy as np
import pandas as pd

import scipy
import matplotlib.pyplot as plt
import urllib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import Parallel, delayed

def KNN(k):
    train = np.loadtxt('trainArray.txt')
    categorias = np.loadtxt("categorias.txt", delimiter='\n', dtype= 'str')
    
    X_train, X_test, y_train, y_test = train_test_split(train, categorias, test_size = 0.2)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

    print(metrics.classification_report(y_test, y_pred))
    print("Metrica de k:" + str(k))

if __name__ == "__main__":
    KNN(3)
    '''
    lista = [1, 3 , 5, 7]
    Parallel(n_jobs=4, verbose=1)(delayed(KNN)(i) for i in lista)
   '''