import matplotlib.pyplot as plt
import numpy as np

# Dados do primeiro conjunto
x1 = np.random.normal(0, 1, 100)
y1 = np.random.normal(0, 1, 100)
z1 = np.random.normal(0, 1, 100)

# Dados do segundo conjunto
x2 = np.random.normal(1, 1, 100)
y2 = np.random.normal(1, 1, 100)
z2 = np.random.normal(1, 1, 100)

# Criar figura e eixos
fig, ax = plt.subplots()

# Scatter plot do primeiro conjunto
sc1 = ax.scatter(x1, y1, c=z1, cmap='viridis', label='Conjunto 1')

# Scatter plot do segundo conjunto
sc2 = ax.scatter(x2, y2, c=z2, cmap='viridis', label='Conjunto 2')

# Adicionar barra de cores
plt.colorbar(sc1, ax=ax, label='Z')

# Rótulos dos eixos
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Legenda
ax.legend()

# Mostrar gráfico
plt.show()
