import numpy as np
from sklearn.neighbors import KernelDensity

class NBGaussianApprox:
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

        var_0_hat = np.maximum(var_0_hat, 1e-9)
        var_1_hat = np.maximum(var_1_hat, 1e-9)

        # priors
        pi_0 = self.n0_ / n
        pi_1 = self.n1_ / n

        self.mu_hat_ = np.array([mu_0_hat, mu_1_hat])
        self.var_hat_ = np.array([var_0_hat, var_1_hat])
        self.pi_ = np.array([pi_0, pi_1])

    # gaussian approx
    def _log_dens_k(self, X, k):
        mu = self.mu_hat_[k]
        var = self.var_hat_[k]

        log_dens_matrix = -0.5 * np.log(2 * np.pi) -0.5 * np.log(var) - (X - mu)**2 / (2 * var)
        log_dens = np.sum(log_dens_matrix, axis=1)

        return log_dens

    def predict_proba(self, Xtest):
        # change for different density estimation
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
    
class NBKDE():
    def __init__(self, bandwidth=0.5):
        self.bandwidth_ = bandwidth

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X_0 = X[y == 0]
        X_1 = X[y == 1]

        n = X.shape[0]

        self.n0_ = X_0.shape[0]
        self.n1_ = X_1.shape[0]

        # priors
        pi_0 = self.n0_ / n
        pi_1 = self.n1_ / n

        self.X_ = [X_0, X_1]
        self.pi_ = np.array([pi_0, pi_1])

    def _log_dens_est_kde(self, Xtest, k):
        Xtest = np.array(Xtest)
        X_k = self.X_[k]

        n_test, p = Xtest.shape
        log_dens = np.zeros(n_test)

        for j in range(p):
            train_feature = X_k[:, j].reshape(-1, 1)

            kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth_)
            kde.fit(train_feature)

            test_feature = Xtest[:, j].reshape(-1, 1)
            log_density_j = kde.score_samples(test_feature)

            log_dens += log_density_j

        return log_dens
    
    def predict_proba(self, Xtest):
        # change for different density estimation
        Xtest = np.array(Xtest)

        log_dens_1 = self._log_dens_est_kde(Xtest=Xtest, k=1)
        log_dens_0 = self._log_dens_est_kde(Xtest=Xtest, k=0)

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
    

class NBDiscretized:
    def __init__(self, n_bins=5):
        self.n_bins_ = n_bins

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n, p = X.shape

        X_0 = X[y == 0]
        X_1 = X[y == 1]

        self.n0_ = X_0.shape[0]
        self.n1_ = X_1.shape[0]

        # priors
        pi_0 = self.n0_ / n
        pi_1 = self.n1_ / n

        self.pi_ = np.array([pi_0, pi_1])

        #### OGARNAC TO ####
        # bin edges
        self.bin_edges_ = []
        X_binned = np.zeros_like(X, dtype=int)

        for j in range(p):
            edges = np.linspace(start=X[:, j].min(), stop=X[:, j].max(), num=self.n_bins_ + 1)
            self.bin_edges_.append(edges)

            # bin numbers: 0, 1, ..., n_bins-1
            X_binned[:, j] = np.digitize(X[:, j], edges[1:-1], right=False)

        # conditional probabilities:
        # self.cond_prob_[k, j, b] = P(feature j in bin b | class k)
        self.cond_prob_ = np.zeros((2, p, self.n_bins_))

        for k in [0, 1]:
            X_k = X_binned[y == k]
            n_k = X_k.shape[0]

            for j in range(p):
                counts = np.bincount(X_k[:, j], minlength=self.n_bins_)

                # Laplace smoothing
                probs = (counts + 1) / (n_k + self.n_bins_)
                self.cond_prob_[k, j, :] = probs

        return self
    
    def _transform_to_bins(self, X):
        X = np.array(X)
        n, p = X.shape
        X_binned = np.zeros_like(X, dtype=int)

        for j in range(p):
            edges = self.bin_edges_[j]
            X_binned[:, j] = np.digitize(X[:, j], edges[1:-1], right=False)

        return X_binned

    def _log_dens_k(self, X, k):
        X_binned = self._transform_to_bins(X)
        n, p = X_binned.shape

        log_dens = np.zeros(n)

        for j in range(p):
            bins_j = X_binned[:, j]
            log_dens += np.log(self.cond_prob_[k, j, bins_j])

        return log_dens
    
    def predict_proba(self, Xtest):
        Xtest = np.array(Xtest)

        log_dens_0 = self._log_dens_k(Xtest, k=0)
        log_dens_1 = self._log_dens_k(Xtest, k=1)

        log_joint_0 = log_dens_0 + np.log(self.pi_[0])
        log_joint_1 = log_dens_1 + np.log(self.pi_[1])

        delta = log_joint_1 - log_joint_0
        posterior_1 = 1 / (1 + np.exp(-delta))

        return posterior_1

    def predict(self, Xtest, thr=0.5):
        posterior_1 = self.predict_proba(Xtest)
        return (posterior_1 >= thr).astype(int)

    
    


        