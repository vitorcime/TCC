import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from numpy import array   
import numpy as np
from multiprocessing import Pool
import sys 



if __name__ == "__main__":
    if (sys.argv[1] == 'train'):
        arquivo = pd.read_csv('../CSV/audiosVerificados.csv')
    if (sys.argv[1] == 'test'):
        arquivo = pd.read_csv('../CSV/test_post_competition.csv')    
        
    print(len(arquivo))
    arquivo = pd.DataFrame(arquivo, columns = ['fname', 'label'])
    arquivo.sort_values(by=['fname'], inplace=True)
    print(arquivo)
    categorias = arquivo['label']
        
        
    np.savetxt('../categorias/categoriasSpecs'+ sys.argv[1] +'.txt', categorias, newline='\n', fmt='%s')
    

