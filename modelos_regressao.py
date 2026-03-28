import numpy as np

class RegressaoLinearMQO:
    def __init__(self, lambd=0.0, fit_intercept=True):
        self.lambd = lambd # Se lambd=0 MQO Tradicional. Se > 0 Regularizado.
        self.fit_intercept = fit_intercept
        self.beta_hat = None
            
    def fit(self, X_train, y_train):
        N, p = X_train.shape
        
        if self.fit_intercept:
            X_train = np.hstack((np.ones((N, 1)), X_train))
            
        # Matriz Identidade para a regularização (dimensão p+1 se tiver intercepto)
        I = np.eye(X_train.shape[1])
        
        # Não aplicamos penalidade no intercepto (índice 0,0 vira 0)
        if self.fit_intercept:
            I[0, 0] = 0 
            
        # Equação Normal com Tikhonov: (X^T * X + lambda * I)^-1 * X^T * y
        self.beta_hat = np.linalg.inv(X_train.T @ X_train + self.lambd * I) @ X_train.T @ y_train
        
    def predict(self, X_test):
        N_test = X_test.shape[0]
        if self.fit_intercept:
            X_test = np.hstack((np.ones((N_test, 1)), X_test))
            
        return X_test @ self.beta_hat

class ModeloMedia:
    def __init__(self):
        self.media_y = None
        
    def fit(self, X_train, y_train):
        # Calcula a média da variável dependente no treino
        self.media_y = np.mean(y_train)
        
    def predict(self, X_test):
        # Retorna um vetor com a mesma quantidade de linhas do X_test, mas preenchido com a média
        N_test = X_test.shape[0]
        return np.full((N_test, 1), self.media_y)