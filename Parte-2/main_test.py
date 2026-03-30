import numpy as np
from classifiers import MyClassifiers

# Carga dos dados
print("Lendo dados...")
# Nota: Certifique-se de que o arquivo está no formato correto (p x N)
data = np.genfromtxt('../assets/EMGsDataset (2).csv', delimiter=',')

# Separação simples para teste (80% treino, 20% teste)
N_train = 40000

# Dados originais parecem estar em (3, 50000)
X_train = data[0:2, :N_train]
y_train = data[2, :N_train].astype(int)

X_test = data[0:2, N_train:]
y_test = data[2, N_train:].astype(int)

def my_one_hot(y, n_classes=5):
    n_samples = len(y)
    Y_oh = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        Y_oh[i, y[i]-1] = 1
    return Y_oh

Y_train_oh = my_one_hot(y_train)

# Execução dos 6 Modelos
clf = MyClassifiers()

print("\n" + "="*40)
print("INICIANDO COMPARAÇÃO DE MODELOS")
print("="*40)

# MQO Tradicional 
clf.fit_mqo(X_train.T, Y_train_oh)
pred_mqo = clf.predict_mqo(X_test.T)
print(f"1. MQO Tradicional:              {np.mean(pred_mqo == y_test)*100:.2f}%")

# Lista de modelos Gaussianos para testar
modelos_gaussianos = [
    ("tradicional", 0, "2. Gaussiano Tradicional (QDA)"),
    ("iguais", 0,      "3. Gaussiano Cov. Iguais (LDA)"),
    ("agregada", 0,    "4. Gaussiano Matriz Agregada  "),
    ("friedman", 0.5,  "5. Gaussiano Regularizado     "),
    ("naive", 0,       "6. Naive Bayes (Diagonal)     ")
]

for m_type, lmbd, label in modelos_gaussianos:
    clf.fit_gaussian(X_train, y_train, model_type=m_type, lmbd=lmbd)
    preds = clf.predict_gaussian(X_test)
    acc = np.mean(preds == y_test)
    print(f"{label}: {acc*100:.2f}%")

print("="*40)