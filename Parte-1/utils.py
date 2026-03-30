import numpy as np
import matplotlib.pyplot as plt
from modelos_regressao import RegressaoLinearMQO, ModeloMedia

def carregar_dados(caminho_arquivo):
    dados = np.loadtxt(caminho_arquivo)
    X = dados[:, 0].reshape(-1, 1) 
    y = dados[:, 1].reshape(-1, 1)
    return X, y

def plotar_dispersao(X, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.5, color='blue', edgecolors='k')
    plt.title('Velocidade do Vento vs. Potência Gerada')
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.grid(True)
    plt.show()

def plotar_predicoes_grid(X, y):
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    mod_media = ModeloMedia()
    mod_media.fit(X, y)
    y_line_media = mod_media.predict(X_line)

    mod_mqo_trad = RegressaoLinearMQO(lambd=0.0)
    mod_mqo_trad.fit(X, y)
    y_line_trad = mod_mqo_trad.predict(X_line)

    mod_mqo_reg = RegressaoLinearMQO(lambd=1.0)
    mod_mqo_reg.fit(X, y)
    y_line_reg = mod_mqo_reg.predict(X_line)

    # Grid 2x2
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análise Visual dos Modelos de Regressão', fontsize=16, fontweight='bold')

    # Apenas os Dados
    axs[0, 0].scatter(X, y, alpha=0.3, edgecolors='k')
    axs[0, 0].set_title('Dados Reais')
    axs[0, 0].set_ylabel('Potência Gerada')
    axs[0, 0].grid(True)

    # Modelo Média
    axs[0, 1].scatter(X, y, alpha=0.3, edgecolors='k')
    axs[0, 1].plot(X_line, y_line_media, color='red', linewidth=3, label='Média')
    axs[0, 1].set_title('Modelo: Média da Variável Dependente')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # MQO Tradicional
    axs[1, 0].scatter(X, y, alpha=0.3, edgecolors='k')
    axs[1, 0].plot(X_line, y_line_trad, color='green', linewidth=3, label='MQO Tradicional')
    axs[1, 0].set_title('Modelo: MQO Tradicional')
    axs[1, 0].set_xlabel('Velocidade do Vento')
    axs[1, 0].set_ylabel('Potência Gerada')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # MQO Regularizado
    axs[1, 1].scatter(X, y, alpha=0.3, edgecolors='k')
    axs[1, 1].plot(X_line, y_line_reg, color='orange', linewidth=3, label='MQO Reg. (λ=1)')
    axs[1, 1].set_title('Modelo: MQO Regularizado (λ=1)')
    axs[1, 1].set_xlabel('Velocidade do Vento')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def calcular_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def calcular_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def executar_monte_carlo(X, y, rodadas=500):
    N_total = X.shape[0]
    qtd_treino = int(0.8 * N_total) # 80% para treno
    lambdas = [0, 0.25, 0.5, 0.75, 1] 
    
    nomes_modelos = [
        'Média da variável dependente',
        'MQO tradicional',
        'MQO regularizado (0,25)',
        'MQO regularizado (0,5)',
        'MQO regularizado (0,75)',
        'MQO regularizado (1)'
    ]
    
    resultados_mse = {nome: [] for nome in nomes_modelos}
    resultados_r2 = {nome: [] for nome in nomes_modelos}

    print(f"Iniciando as {rodadas} rodadas de Random Subsampling Validation... ")
    
    for _ in range(rodadas):
        # Particionamento dos dados em treino e teste
        indices = np.random.permutation(N_total)
        X_train, y_train = X[indices[:qtd_treino]], y[indices[:qtd_treino]]
        X_test, y_test = X[indices[qtd_treino:]], y[indices[qtd_treino:]]
        
        # Modelo de Média 
        mod_media = ModeloMedia()
        mod_media.fit(X_train, y_train)
        y_pred_media = mod_media.predict(X_test)
        resultados_mse['Média da variável dependente'].append(calcular_mse(y_test, y_pred_media))
        resultados_r2['Média da variável dependente'].append(calcular_r2(y_test, y_pred_media))
        
        # Modelos MQO Tradicional e Regularizado
        for lbd in lambdas:
            mod_mqo = RegressaoLinearMQO(lambd=lbd)
            mod_mqo.fit(X_train, y_train)
            y_pred_mqo = mod_mqo.predict(X_test)
            
            if lbd == 0:
                nome_modelo = 'MQO tradicional'
            else:
                nome_modelo = f'MQO regularizado ({str(lbd).replace(".", ",")})'
                
            resultados_mse[nome_modelo].append(calcular_mse(y_test, y_pred_mqo))
            resultados_r2[nome_modelo].append(calcular_r2(y_test, y_pred_mqo))

    return resultados_mse, resultados_r2

def imprimir_tabela(titulo, dicionario_resultados):
    print(f"\n{titulo}")
    print(f"{'Modelos':<30} | {'Média':<10} | {'Desvio-Padrão':<15} | {'Maior Valor':<12} | {'Menor Valor':<12}")
    print("-" * 88)
    for modelo, lista in dicionario_resultados.items():
        media = np.mean(lista)
        desvio = np.std(lista)
        maior = np.max(lista)
        menor = np.min(lista)
        print(f"{modelo:<30} | {media:<10.4f} | {desvio:<15.4f} | {maior:<12.4f} | {menor:<12.4f}")