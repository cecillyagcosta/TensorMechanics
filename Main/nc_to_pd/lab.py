import matplotlib.pyplot as plt
import numpy as np

# Dados
x1 = np.random.normal(size=100)
y1 = np.random.normal(size=100)
z1 = np.random.normal(size=100)

x2 = np.random.normal(size=100)
y2 = np.random.normal(size=100)
z2 = np.random.normal(size=100)

# Criar figura e eixos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot 1
sc1 = ax1.scatter(x1, y1, c=z1, cmap='viridis')
ax1.set_xlabel('X1')
ax1.set_ylabel('Y1')
ax1.set_title('Scatter Plot 1')
fig.colorbar(sc1, ax=ax1, label='Z1')

# Scatter plot 2
sc2 = ax2.scatter(x2, y2, c=z2, cmap='viridis')
ax2.set_xlabel('X2')
ax2.set_ylabel('Y2')
ax2.set_title('Scatter Plot 2')
fig.colorbar(sc2, ax=ax2, label='Z2')

# Ajustar espaçamento entre subplots
plt.tight_layout()

# Mostrar gráfico
plt.show()
