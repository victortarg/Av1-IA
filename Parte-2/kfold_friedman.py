import numpy as np
import matplotlib.pyplot as plt
from classifiers import MyClassifiers

data = np.genfromtxt('../assets/EMGsDataset (2).csv', delimiter=',')
X = data[0:2, :] 
y = data[2, :].astype(int)

# Normalizacao para evitar matriz singular e garantir convergencia
X = (X - X.min(axis=1).reshape(-1,1)) / (X.max(axis=1).reshape(-1,1) - X.min(axis=1).reshape(-1,1))

lambdas = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K = 5
N = X.shape[1]

# Embaralhar indices antes do K-fold
indices = np.random.permutation(N)
folds = np.array_split(indices, K)

acc_por_lambda = []

print(f"Iniciando K-fold (K={K}) para otimizacao de Lambda...")

for lmbd in lambdas:
    acc_fold = []
    for k in range(K):
        # Definicao de indices de treino e teste do fold
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(K) if i != k])
        
        X_train, y_train = X[:, train_idx], y[train_idx]
        X_test, y_test = X[:, test_idx], y[test_idx]
        
        clf = MyClassifiers()
        clf.fit_gaussian(X_train, y_train, model_type="friedman", lmbd=lmbd)
        preds = clf.predict_gaussian(X_test)
        
        acc_fold.append(np.mean(preds == y_test))
    
    media_acc = np.mean(acc_fold)
    acc_por_lambda.append(media_acc)
    print(f"Lambda: {lmbd:<6} | Acuracia Media: {media_acc*100:.2f}%")

# Identificar o melhor lambda
melhor_idx = np.argmax(acc_por_lambda)
print(f"\nMelhor Lambda encontrado: {lambdas[melhor_idx]} com {acc_por_lambda[melhor_idx]*100:.2f}% de acuracia")

# Gerar o grafico para o Overleaf
plt.figure(figsize=(10, 6))
plt.plot(lambdas, acc_por_lambda, marker='o', color='#8B0000', linestyle='--', linewidth=2)
plt.title('K-Fold Cross Validation: Otimizacao de Lambda (Modelo de Friedman)')
plt.xlabel('Parametro de Regularizacao (Lambda)')
plt.ylabel('Acuracia Media (5 Folds)')
plt.grid(True, alpha=0.3)
plt.savefig('kfold_result.png', dpi=300)
plt.show()