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
    import os
    import numpy as np
    import time

    from scipy.special import digamma, gammaln, hyp1f1, polygamma
    from numpy.matlib import repmat
    from sklearn.cluster import KMeans
    izip = zip

    from config import LOG_DIR
    from utils import cluster_acc, predict, log_normalize, caculate_pi, calculate_mix, d_hyp1f1, s_kmeans
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
        self.max_hy1f1_iter = args.max_hy1f1_iter
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
            'zeta': self.args.z,
            'u': self.args.u,
            'v': self.args.v,
            'gamma': self.args.gamma,
        }

        self.u = np.ones(self.T) * self.prior['u']
        self.v = np.ones(self.T) * self.prior['v']
        self.zeta = np.ones(self.T)
        self.xi = np.ones((self.T, self.D))
        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        kmeans = KMeans(n_clusters=self.T).fit(data)
        self.rho = repmat(caculate_pi(kmeans.labels_, self.N, self.T), self.N, 1)
        # self.rho = np.ones((self.N, self.T)) * (1 / self.T)
        self.g = np.zeros(self.T)
        self.h = np.zeros(self.T)

        self.update_zeta_xi(data, self.rho)
        self.update_u_v(self.rho)
        self.update_g_h(self.rho)

    def caclulate_log_lik_x(self, x):

        D = self.D
        E_k = digamma(self.u) - digamma(self.u + self.v)
        kdk1 = d_hyp1f1(0.5, D / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter)
        kdk2 = d_hyp1f1(1.5, (D + 2) / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter) * kdk1
        kdk3 = d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)
        temp = (1 / D * kdk1 + self.zeta * self.k * (
                3 / ((D + 2) * D) * kdk2 - (1 / (D ** 2)) * kdk1 * kdk1)) * self.k * (
                       E_k + np.log(self.zeta) - np.log(self.prior['zeta'] * self.k))
        log_lik_x = gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) + (D / 2) * E_k - np.log(
            (self.k ** (D / 2)) * hyp1f1(0.5, D / 2, self.k)) - (D / 2 / self.k + 1 / D * kdk3) * (
                           self.u / self.v - self.k) + self.k / D * kdk1 + temp * (
                           x.dot(self.xi.T) ** 2)
        return log_lik_x

    def var_inf(self, x):

        begin = time.time()
        for ite in range(self.args.max_iter):
            # compute rho
            E_log_1_pi = np.roll(np.cumsum(digamma(self.h) - digamma(self.g + self.h)), 1)
            E_log_1_pi[0] = 0

            self.rho = self.caclulate_log_lik_x(x) + digamma(self.g) - digamma(self.g + self.h) + E_log_1_pi

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
                times = time.time() - begin
                logger = open(os.path.join(LOG_DIR, "log_times_0.txt"), 'a')
                logger.write(
                    'nyu: times: {}\n'.format(times)
                )
                logger.close()
                self.k = self.u / self.v
                self.k[self.k > self.max_k] = self.max_k
                self.pi = calculate_mix(self.g, self.h, self.T)
                self.calculate_new_com()
                if self.args.verbose:
                    print('mu: {}'.format(self.xi))
                    print('k: {}'.format(self.k))
                    print('pi: {}'.format(self.pi))
                    print('times: {}'.format(times))
                    print('lb: {}'.format(lb))

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
        self.u = self.prior['u'] + (D / 2) * (1 + np.sum(rho, 0)) + self.zeta * self.k / D * d_hyp1f1(0.5, D / 2,
                                                                                                      self.zeta * self.k,
                                                                                                      iteration=self.max_hy1f1_iter)
        self.v = self.prior['v'] + np.sum(rho, 0) * (
                    D / (2 * self.k) + (1 / D) * d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)) + \
                 (D / (2 * self.k) + (zeta / D) * d_hyp1f1(0.5, D / 2, zeta * self.k, iteration=self.max_hy1f1_iter))

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

    def lower_bound(self, x):
        lb = 0
        T = self.T
        D = self.D
        gamma = np.zeros((T, 2))
        gamma[:, 0] = self.g
        gamma[:, 1] = self.h
        sd = digamma(gamma[:, 0] + gamma[:, 1])
        dg0 = digamma(gamma[:, 0]) - sd
        dg1 = digamma(gamma[:, 1]) - sd
        # V
        alpha = self.prior['gamma']
        # Eq[log p(V | 1, alpha)]
        lpv = (gammaln(1 + alpha) - gammaln(alpha)) * T \
              + (alpha - 1) * np.sum(dg1)
        # Eq[log q(V | gamma1, gamma2)]
        lqv = np.sum(gammaln(gamma[:, 0] + gamma[:, 1]) \
                     - gammaln(gamma[:, 0]) - gammaln(gamma[:, 1]) \
                     + (gamma[:, 0] - 1) * dg0 + (gamma[:, 1] - 1) * dg1)
        lb += lpv - lqv

        # mu
        # Eq[log p(u|lambda)]
        kdk1 = d_hyp1f1(0.5, D / 2, self.prior['zeta'] * self.k, iteration=self.max_hy1f1_iter)
        kdk2 = d_hyp1f1(0.5, D / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter)
        lpmu = np.sum(gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) - kdk1 + self.k * (
                    self.xi.dot(self.prior['mu'][:, np.newaxis]).reshape(-1) ** 2))
        # Eq[log q(u|lambda)]
        lqmu = np.sum(gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) - kdk2 + self.k * (np.sum(self.xi ** 2, 1) ** 2))
        lb += lpmu - lqmu

        # kappa
        # Eq[log p(kappa)]
        u, v = self.prior['u'], self.prior['v']
        lpk = np.sum(
            u * np.log(v) - gammaln(u) + (u - 1) * (digamma(self.u) - digamma(self.u + self.v)) - v * (self.u / self.v))
        # Eq[log q(kappa | u, v]
        lqk = np.sum(self.u * np.log(self.v) - gammaln(self.u) + (self.u - 1) * (
                    digamma(self.u) - digamma(self.u + self.v)) - self.v * (self.u / self.v))
        lb += lpk - lqk

        # c
        # Eq[log p(Z | V)]
        pi = calculate_mix(self.g, self.h, self.T)
        lpc = np.sum(self.rho * np.log(pi))
        n_phi = self.rho
        # Eq[log q(Z | phi)]
        lqc = np.sum(n_phi * np.log(n_phi))
        lb += lpc - lqc

        # x
        # Eq[log p(X)]
        likx = self.caclulate_log_lik_x(x)
        lpx = np.sum(self.rho * likx)

        lb += lpx

        return lb

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

        kmeans = KMeans(n_clusters=self.T).fit(data)
        self.rho = repmat(caculate_pi(kmeans.labels_, self.N, self.T), self.N, 1)

        self.prior = {
            'mu': np.sum(data, 0) / np.linalg.norm(np.sum(data, 0)),
            'zeta':self.args.z,
            'u': self.args.u,
            'v': self.args.v,
            'gamma': self.args.gamma,
        }

        self.u = np.ones(self.T) * self.prior['u']
        self.v = np.ones(self.T) * self.prior['v']
        self.zeta = np.ones(self.T)
        self.xi = np.ones((self.T, self.D))
        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.update_zeta_xi(data, self.rho)
        self.update_u_v(self.rho)

    def caclulate_log_lik_x(self, x):

        D = self.D
        E_k = digamma(self.u) - digamma(self.u + self.v)
        kdk1 = d_hyp1f1(0.5, D / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter)
        kdk2 = d_hyp1f1(1.5, (D + 2) / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter) * kdk1
        kdk3 = d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)
        temp = (1 / D * kdk1 + self.zeta * self.k * (
                3 / ((D + 2) * D) * kdk2 - (1 / (D ** 2)) * kdk1 * kdk1)) * self.k * (
                       E_k + np.log(self.zeta) - np.log(self.prior['zeta'] * self.k))
        log_like_x = gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) + (D / 2) * E_k - np.log(
            (self.k ** (D / 2)) * hyp1f1(0.5, D / 2, self.k)) - (D / 2 / self.k + 1 / D * kdk3) * (
                      self.u / self.v - self.k) + self.k / D * kdk1 + temp * (
                      x.dot(self.xi.T) ** 2)
        return log_like_x

    def compute_rho(self, x):

        gamma = self.prior['gamma']
        log_like_x = self.caclulate_log_lik_x(x)
        # collapsed
        E_Nc_minus_n = np.sum(self.rho, 0, keepdims=True) - self.rho
        E_Nc_minus_n_cumsum_geq = np.fliplr(np.cumsum(np.fliplr(E_Nc_minus_n), axis=1))
        E_Nc_minus_n_cumsum = E_Nc_minus_n_cumsum_geq - E_Nc_minus_n

        # var_not_i = np.sum(self.rho * (1 - self.rho), 0, keepdims=True) - self.rho * (1 - self.rho)
        # var_not_i_eq_k = np.zeros((self.N, self.T))
        # for t in range(self.T):
        #     if t != 0:
        #         var_not_i_eq_k[:, t] = np.sum(E_Nc_minus_n[:, :t], 1)
        # var_not_i_eq_k = var_not_i_eq_k * E_greater_i
        # rho += (np.log(1 + E_Nc_minus_n) - var_not_i / (2 * ((1 + E_Nc_minus_n) ** 2))) + (
        #             np.log(gamma + E_greater_i) - var_not_i_eq_k / (2 * ((gamma + E_greater_i) ** 2))) + np.log(
        #     1 + gamma + E_Nc_minus_n + E_greater_i)

        first_tem = np.log(1 + E_Nc_minus_n) - np.log(1 + gamma + E_Nc_minus_n_cumsum_geq)
        first_tem[:, self.T-1] = 0
        dummy = np.log(gamma + E_Nc_minus_n_cumsum) - np.log(1 + gamma + E_Nc_minus_n_cumsum_geq)
        second_term = np.cumsum(dummy, axis=1) - dummy
        rho = log_like_x + (first_tem + second_term)

        log_rho, log_n = log_normalize(rho)
        rho = np.exp(log_rho)
        return rho

    def var_inf(self, x):

        begin = time.time()
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
                times = time.time() - begin
                logger = open(os.path.join(LOG_DIR, "log_times_1.txt"), 'a')
                logger.write(
                    'nyu: times: {}\n'.format(times)
                )
                logger.close()
                self.k = self.u / self.v
                self.k[self.k > self.max_k] = self.max_k
                if self.args.verbose:
                    print('mu: {}'.format(self.xi))
                    print('k: {}'.format(self.k))
                    print('times: {}'.format(times))

    def update_u_v(self, rho):

        D = self.D
        zeta = self.prior['zeta']
        # compute u, v
        self.u = self.prior['u'] + (D / 2) * (1 + np.sum(rho, 0)) + self.zeta * self.k / D * d_hyp1f1(0.5, D / 2,
                                                                                                      self.zeta * self.k,
                                                                                                      iteration=self.max_hy1f1_iter)
        self.v = self.prior['v'] + np.sum(rho, 0) * (
                    D / (2 * self.k) + (1 / D) * d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)) + \
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

    def lower_bound(self, x):
        lb = 0
        D = self.D
        # mu
        # Eq[log p(u|lambda)]
        kdk1 = d_hyp1f1(0.5, D / 2, self.prior['zeta'] * self.k, iteration=self.max_hy1f1_iter)
        kdk2 = d_hyp1f1(0.5, D / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter)
        lpmu = np.sum(gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) - kdk1 + self.k * (
                    self.xi.dot(self.prior['mu'][:, np.newaxis]).reshape(-1) ** 2))
        # Eq[log q(u|lambda)]
        lqmu = np.sum(gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) - kdk2 + self.k * (np.sum(self.xi ** 2, 1) ** 2))
        lb += lpmu - lqmu

        # kappa
        # Eq[log p(kappa)]
        u, v = self.prior['u'], self.prior['v']
        lpk = np.sum(
            u * np.log(v) - gammaln(u) + (u - 1) * (digamma(self.u) - digamma(self.u + self.v)) - v * (self.u / self.v))
        # Eq[log q(kappa | u, v]
        lqk = np.sum(self.u * np.log(self.v) - gammaln(self.u) + (self.u - 1) * (
                    digamma(self.u) - digamma(self.u + self.v)) - self.v * (self.u / self.v))
        lb += lpk - lqk

        # c
        # Eq[log p(Z | V)]
        lpc = np.sum(self.rho)
        n_phi = self.rho
        # Eq[log q(Z | phi)]
        lqc = np.sum(n_phi * np.log(n_phi))
        lb += lpc - lqc

        # x
        # Eq[log p(X)]
        likx = self.caclulate_log_lik_x(x)
        lpx = np.sum(self.rho * likx)

        lb += lpx

        return lb

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
