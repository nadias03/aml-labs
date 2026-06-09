import numpy as np

class NB:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X_0 = X[y == 0]
        X_1 = X[y == 1]

        n = X.shape[0]

        self.n0_ = X_0.shape[0]
        self.n1_ = X_1.shape[0]

        # vectors of means
        mu_0_hat = np.mean(X_0, axis=0)
        mu_1_hat = np.mean(X_1, axis=0)

        # in nb we assume conditional independence - variance estimation (instead of covariance)
        var_0_hat = np.var(X_0, axis=0)
        var_1_hat = np.var(X_1, axis=0)

        var_0_hat = np.maximum(var_0_hat, 1e-9)   # max zeby uniknac zbyt malej wartosci wariancji
        var_1_hat = np.maximum(var_1_hat, 1e-9)

        # priors
        pi_0 = self.n0_ / n
        pi_1 = self.n1_ / n

        self.mu_hat_ = np.array([mu_0_hat, mu_1_hat])
        self.var_hat_ = np.array([var_0_hat, var_1_hat])
        self.pi_ = np.array([pi_0, pi_1])

    def _log_dens_k(self, X, k):
        mu = self.mu_hat_[k]
        var = self.var_hat_[k]

        log_dens_matrix = -0.5 * np.log(2 * np.pi) -0.5 * np.log(var) - (X - mu)**2 / (2 * var)
        log_dens = np.sum(log_dens_matrix, axis=1)

        return log_dens

    def predict_proba(self, Xtest):
        Xtest = np.array(Xtest)

        log_dens_1 = self._log_dens_k(X=Xtest, k=1)
        log_dens_0 = self._log_dens_k(X=Xtest, k=0)

        log_prior_1 = np.log(self.pi_[1])
        log_prior_0 = np.log(self.pi_[0])

        log_joint_prob_1 = log_dens_1 + log_prior_1
        log_joint_prob_0 = log_dens_0 + log_prior_0

        delta = log_joint_prob_1 - log_joint_prob_0

        posterior_1 = 1 / (1 + np.exp(-delta))

        return posterior_1  
    
    def predict(self, Xtest, thr = 0.5):
        posterior_1 = self.predict_proba(Xtest=Xtest)
        return (posterior_1 >= thr).astype(int)
    
    def get_params(self):
        return [
            self.mu_hat_[0],
            self.mu_hat_[1],
            self.var_hat_[0],
            self.var_hat_[1],
            self.pi_[0],
            self.pi_[1],
        ]