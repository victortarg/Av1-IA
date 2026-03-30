import numpy as np
from classifiers import MyClassifiers

# CARGA E NORMALIZAÇÃO
data = np.genfromtxt('../assets/EMGsDataset (2).csv', delimiter=',')
X_raw = data[0:2, :] 
y_raw = data[2, :].astype(int)

# Normalização Min-Max (0-1) - essencial para estabilidade nas 500 rodadas
x_min, x_max = X_raw.min(axis=1).reshape(-1,1), X_raw.max(axis=1).reshape(-1,1)
X_norm = (X_raw - x_min) / (x_max - x_min)

# CONFIGURAÇÃO MONTE CARLO
R = 500
N = X_norm.shape[1]
lmbd_otimo = 0.01 # Valor que encontramos no K-fold


modelos = ["MQO", "Tradicional", "Iguais", "Agregada", "Naive", "Friedman"]
resultados = {m: [] for m in modelos}

print(f"Iniciando Simulação de Monte Carlo: {R} rodadas...")

for r in range(R):
    # Particionamento Aleatório 80/20 (Sorteio manual conforme regra)
    indices = np.random.permutation(N)
    train_idx = indices[:int(0.8 * N)]
    test_idx = indices[int(0.8 * N):]
    
    X_tr, y_tr = X_norm[:, train_idx], y_raw[train_idx]
    X_te, y_te = X_norm[:, test_idx], y_raw[test_idx]
    
    clf = MyClassifiers()
    
    # MQO (Exige N x p e One-Hot)
    Y_tr_oh = np.zeros((len(y_tr), 5))
    for i in range(len(y_tr)): Y_tr_oh[i, y_tr[i]-1] = 1
    
    clf.fit_mqo(X_tr.T, Y_tr_oh)
    acc_mqo = np.mean(clf.predict_mqo(X_te.T) == y_te)
    resultados["MQO"].append(acc_mqo)
    
    # Modelos Gaussianos (p x N)
    tipos = [
        ("tradicional", 0, "Tradicional"),
        ("iguais", 0, "Iguais"),
        ("agregada", 0, "Agregada"),
        ("naive", 0, "Naive"),
        ("friedman", lmbd_otimo, "Friedman")
    ]
    
    for m_type, l, tag in tipos:
        clf.fit_gaussian(X_tr, y_tr, model_type=m_type, lmbd=l)
        acc = np.mean(clf.predict_gaussian(X_te) == y_te)
        resultados[tag].append(acc)
        
    if (r + 1) % 50 == 0:
        print(f"Rodada {r+1}/{R} concluída...")

print("\n" + "="*70)
print(f"{'MODELO':<15} | {'MÉDIA (%)':<10} | {'DESVIO':<10} | {'MÁX (%)':<10} | {'MÍN (%)':<10}")
print("-" * 70)

for m in modelos:
    accs = np.array(resultados[m]) * 100
    print(f"{m:<15} | {np.mean(accs):>9.2f} | {np.std(accs):>9.4f} | {np.max(accs):>9.2f} | {np.min(accs):>9.2f}")
print("="*70)