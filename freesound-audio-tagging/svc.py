import numpy as np
import pandas as pd

import scipy
import matplotlib.pyplot as plt
import urllib
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import Parallel, delayed

def SVM(c):
    train = np.loadtxt('trainArray.txt')
    categorias = np.loadtxt("categorias.txt", delimiter='\n', dtype= 'str')
    
    X_train, X_test, y_train, y_test = train_test_split(train, categorias, test_size = 0.2)
    neigh = SVC(C=c, gamma='auto')
    neigh.fit(X_train, y_train)
    y_expect = y_test
    y_pred = neigh.predict(X_test)

    print(metrics.classification_report(y_expect, y_pred))
    print("Metrica de c:" + str(c))

if __name__ == "__main__":
    lista = [1, 10 , 100, 1000]
    Parallel(n_jobs=4, verbose=1)(delayed(SVM)(i) for i in lista)
   