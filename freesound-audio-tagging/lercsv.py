import pandas as pd

def isThere(objeto, procurado):
    y = 0
    for i in objeto:
        if procurado == i.getCat():
            return y
        y+=1    
    return -1

class Categoria:
    def __init__(self, cat, numero):
        self.cat = cat
        self.numero = numero
    
    def getCat(self):
        return self.cat

    def getNum(self):
        return self.numero

    def setNum(self):
        self.numero+=1

if __name__ == "__main__":

    categorias = []
    readCSV = pd.read_csv("train.csv")
    
    for row in readCSV['label']:
        indice = isThere(categorias, row)
        if indice != -1:
            categorias[indice].setNum()
        else:
            categorias.append(Categoria(row, 1))
    
    arq = open('audiosEscolhidos.txt', 'w')
    readCSV = pd.read_csv("train.csv")
    for categoria in categorias:

       nomes = readCSV['fname']
       label = readCSV['label']
       verificado = readCSV['manually_verified']
       escolhidos = 0
       for i in range(0, len(nomes)):
       
           if(label[i] == categoria.getCat() and verificado[i] == 1):
               if escolhidos >= int(categoria.getNum()*0.1):
                   break
               arq.write(nomes[i] + ", " + label[i] + "\n")
               escolhidos+=1

