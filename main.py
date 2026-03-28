import utils

if __name__ == "__main__":
    print("TRABALHO DE IA: REGRESSÃO")
    
    #Organização dos dados e visualização inicial
    X, y = utils.carregar_dados('./assets/aerogerador (1).dat')
    
    utils.plotar_predicoes_grid(X, y)
    
    # Treinamento e Validação (500 rodadas)
    resultados_mse, resultados_r2 = utils.executar_monte_carlo(X, y, rodadas=500)
    
    # Exibição das tabelas de resultados
    utils.imprimir_tabela("TABELA DE DESEMPENHO: MSE", resultados_mse)
    utils.imprimir_tabela("TABELA DE DESEMPENHO: R²", resultados_r2)
    
    print("\nFIM")