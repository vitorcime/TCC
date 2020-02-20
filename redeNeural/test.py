import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print("Carregando imagens")
imagens_treino = np.load("lista.npy")
print(imagens_treino.shape)
imagens_treino = np.mean(imagens_treino, axis=-1, keepdims=True)
print(imagens_treino.shape)
imagens_treino = scaler.fit((imagens_treino, imagens_treino.shape))
imagens_treino = scaler.transform(imagens_treino)
print(imagens_treino)
