import numpy as np

class LDA:
    def __init__(self, reg=0):
         self.lambda_ = reg

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X_0 = X[y == 0]
        X_1 = X[y == 1]

        n = X.shape[0]
        p = X.shape[1]
        self.p_ = p

        n_0 = len(X_0)
        n_1 = len(X_1)

        # vector of means
        mu_0_hat = np.mean(X_0, axis=0)
        mu_1_hat = np.mean(X_1, axis=0)

        # covariance matrix
        Sigma_hat = np.zeros((p, p))
        X_0c = X_0 - mu_0_hat
        X_1c = X_1 - mu_1_hat
        Sigma_hat += X_0c.T @ X_0c
        Sigma_hat += X_1c.T @ X_1c
        Sigma_hat /= (n - 2)
        Sigma_hat += self.lambda_ * np.eye(p)

        # prior probabilities
        pi_0 = n_0 / n
        pi_1 = n_1 / n

        self.mu_hat_ = np.array([mu_0_hat, mu_1_hat])
        self.Sigma_hat_ = Sigma_hat
        self.pi_ = np.array([pi_0, pi_1])

    def _log_dens_k(self, X, k):
        mu = self.mu_hat_[k]
        Sigma = self.Sigma_hat_
        p = self.p_

        _, Sigma_logdet = np.linalg.slogdet(Sigma)
        X_c = X - mu

        sol = np.linalg.solve(Sigma, X_c.T).T
        quad = np.sum(X_c * sol, axis=1)
        log_dens = -0.5 * p * np.log(2 * np.pi) -0.5 * Sigma_logdet -0.5 * quad

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
    
    def boundary_line(self, X):
        log_dens_1 = self._log_dens_k(X=X, k=1)
        log_dens_0 = self._log_dens_k(X=X, k=0)

        delta_01 = log_dens_0 - log_dens_1 + np.log(self.pi_[0] / self.pi_[1])

        return delta_01

    def predict(self, Xtest, thr = 0.5):
        posterior_1 = self.predict_proba(Xtest=Xtest)
        return (posterior_1 >= thr).astype(int)

    def get_params(self):
        return [
            self.mu_hat_[0], 
            self.mu_hat_[1], 
            self.Sigma_hat_,
            self.pi_[0],
            self.pi_[1],
        ]