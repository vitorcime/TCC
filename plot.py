import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('resultados.csv')
data.columns = ['patchs', 'limiar', 'resultado']


def scatterplot(df, x_dim, y_dim):
  x = df[x_dim]
  y = df[y_dim]
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.scatter(x, y)
  plt.show()
scatterplot(data, 'limiar', 'resultado')