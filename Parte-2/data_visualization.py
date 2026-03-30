import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('../assets/EMGsDataset (2).csv', delimiter=',')
X = data[0:2, :]  # Sensores (p=2)
y = data[2, :].astype(int)  # Categorias (C=5)

classes = ['Neutro', 'Sorriso', 'Sobr. Levantada', 'Surpreso', 'Rabugento']
colors = ['#444444', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

plt.figure(figsize=(12, 8))

for i in range(1, 6):
    mask = (y == i)
    plt.scatter(X[0, mask], X[1, mask], 
                c=colors[i-1], label=classes[i-1], 
                alpha=0.6, s=25, 
                edgecolors='w', linewidths=0.5)

plt.title('Distribuicao de Sinais EMG por Expressao Facial')
plt.xlabel('Sensor 1: Corrugador do Supercilio (Atividade ADC)')
plt.ylabel('Sensor 2: Zigomatico Maior (Atividade ADC)')
plt.legend(loc='upper right', markerscale=1.5, fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)


plt.savefig('fig1_destacada.png', dpi=300)
print("Grafico gerado e salvo como fig1_destacada.png")
plt.show()