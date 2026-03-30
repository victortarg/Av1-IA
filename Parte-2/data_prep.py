import numpy as np

# Linha 0: Sensor 1 | Linha 1: Sensor 2 | Linha 2: Categoria
data = np.genfromtxt('../assets/EMGsDataset (2).csv', delimiter=',')

# Variáveis globais conforme o enunciado
N = 50000
p = 2
C = 5

# MQO
# X deve ser N x p (50000, 2)
X_mqo = data[0:2, :].T 

# Y deve ser N x C (50000, 5) - One-Hot Encoding
y_raw = data[2, :].astype(int)
Y_mqo = np.zeros((N, C))
for i in range(N):
    Y_mqo[i, y_raw[i]-1] = 1

# GAUSSIANOS
# X deve ser p x N (2, 50000)
X_gauss = data[0:2, :]

# Y deve ser C x N (5, 50000)
Y_gauss = np.zeros((C, N))
for i in range(N):
    Y_gauss[y_raw[i]-1, i] = 1

print(f"MQO: X {X_mqo.shape}, Y {Y_mqo.shape}")
print(f"Gaussiano: X {X_gauss.shape}, Y {Y_gauss.shape}")