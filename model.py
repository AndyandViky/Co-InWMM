# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: model.py
@Time: 2020-02-14 00:05
@Desc: model.py
"""
try:
    import numpy as np

    from scipy.special import digamma, gammaln, hyp1f1, polygamma
    from numpy.matlib import repmat
    from sklearn.cluster import KMeans
    izip = zip

    from utils import cluster_acc, predict, log_normalize, caculate_pi, calculate_mix, d_hyp1f1
except ImportError as e:
    print(e)
    raise ImportError


class VIModel:
    """
    Variational Inference Dirichlet process Mixture Models of Watson Distributions
    """

    def __init__(self, args):

        self.T = args.T
        self.max_k = 700
        self.args = args
        self.N = 300
        self.D = 3
        self.prior = dict()
        self.pi = None
        self.newJ = args.T

        self.gamma = None
        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None

        self.rho = None
        self.g = None
        self.h = None

        self.temp_zeta = None
        self.det = 1e-10

    def init_params(self, data):

        (self.N, self.D) = data.shape

        self.prior = {
            'mu': np.sum(data, 0) / np.linalg.norm(np.sum(data, 0)),
            'zeta': 0.02,
            'u': 1,
            'v': 0.01,
            'gamma': 1,
        }
        mu = self.prior['mu'][np.newaxis, :]
        self.prior['zeta'] = np.max(np.linalg.eig(mu.T.dot(mu))[0])

        self.u = np.ones(self.T) * self.prior['u']
        self.v = np.ones(self.T) * self.prior['v']
        self.zeta = np.ones(self.T)
        self.xi = np.ones((self.T, self.D))
        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        kmeans = KMeans(n_clusters=self.T, max_iter=100).fit(data)
        self.rho = repmat(caculate_pi(kmeans, self.N, self.T), self.N, 1)
        # self.rho = np.ones((self.N, self.T)) * (1 / self.T)
        self.g = np.zeros(self.T)
        self.h = np.zeros(self.T)

        self.update_zeta_xi(data, self.rho)
        self.update_u_v(self.rho)
        self.update_g_h(self.rho)

    def var_inf(self, x):

        D = self.D
        for ite in range(self.args.max_iter):
            # compute rho
            E_log_1_pi = np.roll(np.cumsum(digamma(self.h) - digamma(self.g + self.h)), 1)
            E_log_1_pi[0] = 0

            E_k = digamma(self.u) - digamma(self.u + self.v)
            kdk1 = d_hyp1f1(0.5, D / 2, self.zeta * self.k)
            kdk2 = d_hyp1f1(1.5, (D + 2) / 2, self.zeta * self.k) * kdk1
            kdk3 = d_hyp1f1(0.5, D / 2, self.k)
            temp = (1 / D * kdk1 + self.zeta * self.k * (
                        3 / ((D + 2) * D) * kdk2 - (1 / (D ** 2)) * kdk1 * kdk1)) * self.k * (
                               E_k + np.log(self.zeta) - np.log(self.prior['zeta'] * self.k))
            self.rho = gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) + (D / 2) * E_k - np.log(
                (self.k ** (D / 2)) * hyp1f1(0.5, D / 2, self.k)) - (D / 2 / self.k + 1 / D * kdk3) * (
                                   self.u / self.v - self.k) + self.k / D * kdk1 + temp * (
                                   x.dot(self.xi.T) ** 2) + digamma(self.g) - digamma(self.g + self.h) + E_log_1_pi

            log_rho, log_n = log_normalize(self.rho)
            self.rho = np.exp(log_rho)

            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi(x, self.rho)
            self.update_u_v(self.rho)
            self.update_g_h(self.rho)

            print(ite)
            if ite == self.args.max_iter - 1:
                self.k = self.u / self.v
                self.k[self.k > self.max_k] = self.max_k
                self.pi = calculate_mix(self.g, self.h, self.T)
                self.calculate_new_com()
                if self.args.verbose:
                    print('mu: {}'.format(self.xi))
                    print('k: {}'.format(self.k))
                    print('pi: {}'.format(self.pi))

    def calculate_new_com(self):

        threshold = self.args.mix_threshold

        index = np.where(self.pi > threshold)[0]
        self.pi = self.pi[self.pi > threshold]
        self.newJ = self.pi.size

        self.xi = self.xi[index]
        self.k = self.k[index]

        if self.args.verbose:
            print("new component is {}".format(self.newJ))

    def update_u_v(self, rho):

        D = self.D
        zeta = self.prior['zeta']
        # compute u, v
        self.u = self.prior['u'] + (D / 2) * (1 + np.sum(rho, 0)) + self.zeta * self.k / D * d_hyp1f1(0.5, D / 2, self.zeta * self.k)
        self.v = self.prior['v'] + np.sum(rho, 0) * (D / (2 * self.k) + (1 / D) * d_hyp1f1(0.5, D / 2, self.k)) + \
                 (D / (2 * self.k) + (zeta / D) * d_hyp1f1(0.5, D / 2, zeta * self.k))

    def update_zeta_xi(self, x, rho):

        # compute zeta, xi
        mu = self.prior['mu'][np.newaxis, :] # 1 * d
        D = self.D
        for t in range(self.T):
            A = self.prior['zeta'] * mu.T.dot(mu) + x.T.dot(rho[:, t:t+1] * x)
            value, vector = np.linalg.eig(A)
            index = np.argmax(value)
            self.zeta[t] = value[index]
            self.xi[t] = vector[:, index]

    def update_g_h(self, rho):
        # compute g, h
        N_k = np.sum(rho, 0)
        self.g = 1 + N_k
        for i in range(self.T):
            if i == self.T - 1:
                self.h[i] = self.prior['gamma']
            else:
                temp = rho[:, i + 1:self.T]
                self.h[i] = self.prior['gamma'] + np.sum(np.sum(temp, 1), 0)

    def fit(self, data):

        self.init_params(data)
        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        pred = predict(data, mu=self.xi, k=self.k, pi=self.pi, n_cluster=self.newJ)
        return pred

    def fit_predict(self, data):

        self.fit(data)
        return self.predict(data)


class CVIModel:
    """
    Collapsed Variational Inference Dirichlet process Mixture Models of Watson Distributions
    """

    def __init__(self, args):

        self.T = args.T
        self.max_k = 700
        self.args = args
        self.max_hy1f1_iter = args.max_hy1f1_iter
        self.N = 300
        self.D = 3
        self.prior = dict()
        self.newJ = args.T

        self.gamma = None
        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None

        self.rho = None

        self.temp_zeta = None
        self.det = 1e-10

    def init_params(self, data):

        (self.N, self.D) = data.shape

        self.prior = {
            'mu': np.sum(data, 0) / np.linalg.norm(np.sum(data, 0)),
            'zeta': 0.02,
            'u': self.args.u,
            'v': self.args.v,
            'gamma': self.args.gamma,
        }
        mu = self.prior['mu'][np.newaxis, :]
        self.prior['zeta'] = np.max(np.linalg.eig(mu.T.dot(mu))[0])

        self.u = np.ones(self.T) * self.prior['u']
        self.v = np.ones(self.T) * self.prior['v']
        self.zeta = np.ones(self.T)
        self.xi = np.ones((self.T, self.D))
        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        kmeans = KMeans(n_clusters=self.T, max_iter=100).fit(data)
        self.rho = repmat(caculate_pi(kmeans, self.N, self.T), self.N, 1)
        # self.rho = np.ones((self.N, self.T)) * (1 / self.T)

        self.update_zeta_xi(data, self.rho)
        self.update_u_v(self.rho)

    def compute_rho(self, x):

        D = self.D
        E_k = digamma(self.u) - digamma(self.u + self.v)
        kdk1 = d_hyp1f1(0.5, D / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter)
        kdk2 = d_hyp1f1(1.5, (D + 2) / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter) * kdk1
        kdk3 = d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)
        temp = (1 / D * kdk1 + self.zeta * self.k * (
                3 / ((D + 2) * D) * kdk2 - (1 / (D ** 2)) * kdk1 * kdk1)) * self.k * (
                       E_k + np.log(self.zeta) - np.log(self.prior['zeta'] * self.k))
        rho = gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) + (D / 2) * E_k - np.log(
            (self.k ** (D / 2)) * hyp1f1(0.5, D / 2, self.k)) - (D / 2 / self.k + 1 / D * kdk3) * (
                           self.u / self.v - self.k) + self.k / D * kdk1 + temp * (
                           x.dot(self.xi.T) ** 2)
        # collapsed
        E_not_i = np.sum(self.rho, 0, keepdims=True) - self.rho
        E_not_i_eq_k = np.zeros((self.N, self.T))
        for t in range(self.T-1):
            E_not_i_eq_k[:, t] = np.sum(E_not_i[:, t + 1:], 1)

        rho += np.log(1 + E_not_i) + np.log(self.prior['gamma'] + E_not_i_eq_k) + np.log(
            1 + self.prior['gamma'] + E_not_i + E_not_i_eq_k)
        log_rho, log_n = log_normalize(rho)
        rho = np.exp(log_rho)
        return rho

    def var_inf(self, x):

        for ite in range(self.args.max_iter):
            # compute rho
            self.rho = self.compute_rho(x)

            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi(x, self.rho)
            self.update_u_v(self.rho)

            print(ite)
            if ite == self.args.max_iter - 1:
                self.k = self.u / self.v
                self.k[self.k > self.max_k] = self.max_k
                if self.args.verbose:
                    print('mu: {}'.format(self.xi))
                    print('k: {}'.format(self.k))

    def update_u_v(self, rho):

        D = self.D
        zeta = self.prior['zeta']
        # compute u, v
        self.u = self.prior['u'] + (D / 2) * (1 + np.sum(rho, 0)) + self.zeta * self.k / D * d_hyp1f1(0.5, D / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter)
        self.v = self.prior['v'] + np.sum(rho, 0) * (D / (2 * self.k) + (1 / D) * d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)) + \
                 (D / (2 * self.k) + (zeta / D) * d_hyp1f1(0.5, D / 2, zeta * self.k, iteration=self.max_hy1f1_iter))

    def update_zeta_xi(self, x, rho):

        # compute zeta, xi
        mu = self.prior['mu'][np.newaxis, :] # 1 * d
        for t in range(self.T):
            A = self.prior['zeta'] * mu.T.dot(mu) + x.T.dot(rho[:, t:t+1] * x)
            value, vector = np.linalg.eig(A)
            index = np.argmax(value)
            self.zeta[t] = value[index]
            self.xi[t] = vector[:, index]

    def fit(self, data):

        self.init_params(data)
        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        rho = self.compute_rho(data)
        return np.argmax(rho, axis=1)

    def fit_predict(self, data):

        self.fit(data)
        return self.predict(data)
