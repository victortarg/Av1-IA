import numpy as np

class MyClassifiers:
    def __init__(self):
        self.W = None 
        self.params = {}

    # MQO COM PSEUDO-INVERSA
    def fit_mqo(self, X, Y):
        # X: (N, p), Y: (N, C)
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        self.W = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ Y

    def predict_mqo(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.argmax(X_b @ self.W, axis=1) + 1

    # GAUSSIANOS COM PREDICÃO OTIMIZADA
    def fit_gaussian(self, X, y, model_type="tradicional", lmbd=0):
        classes = np.unique(y)
        p = X.shape[0]
        mus, sigmas, prioris = [], [], []

        for c in classes:
            X_c = X[:, y == c]
            mus.append(np.mean(X_c, axis=1).reshape(-1, 1))
            sigmas.append(np.cov(X_c))
            prioris.append(X_c.shape[1] / len(y))

        sigma_pool = np.mean(sigmas, axis=0)
        final_sigmas = []

        for i, S_i in enumerate(sigmas):
            if model_type == "tradicional": S = S_i
            elif model_type == "iguais": S = sigma_pool
            elif model_type == "agregada": S = 0.5 * S_i + 0.5 * sigma_pool
            elif model_type == "naive": S = np.diag(np.diag(S_i))
            elif model_type == "friedman": S = (1 - lmbd) * S_i + lmbd * sigma_pool
            
            final_sigmas.append(S + 1e-6 * np.eye(p))

        self.params = {'mus': mus, 'sigmas': final_sigmas, 'prioris': prioris, 'classes': classes}

    def predict_gaussian(self, X):
        classes = self.params['classes']
        n_classes = len(classes)
        N_samples = X.shape[1]
        
        # Pré-calculo com pinv para ser rápido e seguro
        inv_sigmas = [np.linalg.pinv(S) for S in self.params['sigmas']]
        log_dets = [np.log(np.linalg.det(S)) for S in self.params['sigmas']]
        log_prioris = [np.log(p) for p in self.params['prioris']]
        mus = self.params['mus']

        preds = []
        for j in range(N_samples):
            x = X[:, j].reshape(-1, 1)
            scores = []
            for i in range(n_classes):
                diff = x - mus[i]
                # Função Discriminante Gaussiana
                term1 = -0.5 * log_dets[i]
                term2 = -0.5 * (diff.T @ inv_sigmas[i] @ diff)
                scores.append(term1 + term2 + log_prioris[i])
            preds.append(classes[np.argmax(scores)])
        return np.array(preds)