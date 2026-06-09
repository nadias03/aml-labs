import numpy as np

class DataGenerator:
    def __init__(self):
        pass

    def scheme1(self, a, n=1000, p=2, suc_prob=0.5):
        y = np.random.binomial(n=1, p=suc_prob, size=n)

        mask_0 = (y == 0)
        mask_1 = (y == 1)

        n0 = mask_0.sum()
        n1 = mask_1.sum()

        X = np.zeros((n, p))

        X0 = np.random.normal(loc=0, scale=1, size=(n0, p))
        X[mask_0] = X0

        X1 = np.random.normal(loc=a, scale=1, size=(n1, p))
        X[mask_1] = X1

        return X, y
    
    def scheme2(self, a, rho, n=1000, p=2, suc_prob=0.5):
        y = np.random.binomial(n=1, p=suc_prob, size=n)

        mask_0 = (y == 0)
        mask_1 = (y == 1)

        n0 = mask_0.sum()
        n1 = mask_1.sum()

        X = np.zeros((n, p))

        mu_0 = np.array([0, 0])
        Sigma_0 = np.array([
            [1, rho],
            [rho, 1]
        ])
        X0 = np.random.multivariate_normal(mean=mu_0, cov=Sigma_0, size=n0)
        X[mask_0] = X0
       
        mu_1 = np.array([a, a])
        Sigma_1 = np.array([
            [1, -rho],
            [-rho, 1]
        ])
        X1 = np.random.multivariate_normal(mean=mu_1, cov=Sigma_1, size=n1)
        X[mask_1] = X1

        return X, y