#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:04:50 2018

@author: WangJianqiao
"""
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

class Deconvolution:
    def __init__(self, Zn):
        '''
        Zn: observed data
        n: sample size
        '''
        self.Zn = Zn
        self.n = len(Zn)

    def _g(self, theta, x):
        '''
        for MLE
        g_theta(x)
        '''
        #return (self.K(x) - self.K(x-theta)) / theta
        return stats.norm.pdf(x, loc=theta)

    # def _F(self, theta, x):
    #     '''
    #     for MLE
    #     F_theta(x)
    #     '''
    #     if x > theta:
    #         return 1
    #     else:
    #         return x / theta

    def _c1(self, theta, coefficient=None):
        '''
        mle: c_1(theta, g_bar), for computing new support
        '''
        c1 = 0

        for zi in self.Zn:
            c1 += self._g(theta, zi) / self._g_bar(zi)

        return 1 - c1 / self.n

    def _c2(self, theta):
        '''
        for MLE
        c2(theta, g_bar) for computing new support
        '''
        c2 = 0
        
        for zi in self.Zn:
            c2 = (self._g(theta, zi) ** 2) / (self._g_bar(zi) ** 2)

        return c2 / self.n

    def _g_bar(self, x):
        '''
        g for current iteration
        '''
        g = 0
        
        for theta in self.coefficient.keys():
            g += self.coefficient[theta] * self._g(theta, x)
        
        return g
    
    def _new_support(self, coefficient):
        '''
        find new support
        '''
        support = set(coefficient.keys())
        # set of new supports
        new_support_set = list(set(self.Zn) - support)
            
        if len(support) == self.n:
            # no new support, algorithm ended
            return False
        
        # find the least c_1(theta, g_bar) / sqrt(c_2(theta))
        support_eval = np.zeros(len(new_support_set))
        
        for i in range(len(new_support_set)):
            c1 = self._c1(new_support_set[i])
            c2 = self._c2(new_support_set[i])
            support_eval[i] = c1 / np.sqrt(c2)

        return new_support_set[np.argmin(support_eval)]
    

    # def _solve(self, support_set):
    #     '''
    #     given a certain support set, solve linear equations to obtain new coefficients
    #     '''
    #     # number of supports
    #     m = len(support_set)
    #     # support set
    #     support = list(support_set)
    #     # coefficient: {theta: alpha}
    #     coefficient = dict()
        
    #     '''
    #     A(alpha_1, ..., alpha_m) = b
    #     '''
    #     A = np.mat(np.zeros((m, m)))
    #     b = np.mat(np.zeros((m, 1)))
        
    #     # obtian A and b
    #     for i in range(m):
    #         for j in range(m):
    #             a = 0
    #             for zi in self.Zn:
    #                 a += self._g(support[i], zi) * self._g(support[j], zi) / (self._g_bar(zi) ** 2)
    #             a = a / self.n

    #             A[i, j] = a
    #         b[i, 0] = 1 - 2 * self._c1(support[i])
        
    #     # coefficients: (alpha_1, ..., alpha_m)
    #     coef = np.matmul(A.I, b)
    #     #coefficient = dict()
        
    #     # sign: denotes whether there is a negative term in coefficients
    #     # sign: negative coefficient exists, False; otherwise, True
    #     sign = True
        
    #     for i in range(m):
    #         coefficient[support[i]] = coef[i, 0]
    #         if coef[i, 0] < 0:
    #             sign = False
        
    #     return coefficient, sign

    def _solve(self, support_set):
        '''
        given a certain support set, solve linear equations to obtain new coefficients
        '''
        support = list(support_set)
        p = len(support_set)
        # Y is a n*p matrix: Y_{ij}=f_{theta_j}(x_i)
        Y = np.array([[self._g(theta_j, zi) for theta_j in support] for zi in self.Zn])
        # d is a n-vector: d_i=(g_bar(x_i))^{-1}
        d = np.array([1 / self._g_bar(zi) for zi in self.Zn])
        # D ia a n*n diagonal matrix with D_ii =d_i
        D = np.diag(d)
        '''
        The linear equations are
        (DY)'DY alpha = 2 Y' d - n_p
        '''
        DY = D.dot(Y)  # n*p
        # coefficients: (alpha_1, ..., alpha_m)
        coef = np.linalg.solve(np.dot(DY.T, DY), 2 * np.dot(Y.T, d) - np.array([self.n] * p))
        coefficient = dict()

        # sign: denotes whether there is a negative term in coefficients
        # sign: negative coefficient exists, False; otherwise, True
        sign = True
        
        for i in range(p):
            coefficient[support[i]] = coef[i, 0]
            if coef[i, 0] < 0:
                sign = False
        
        return coefficient, sign
        
        
    
    def _remove_index(self, old_coefficient, new_coefficient):
        '''
        support needed to be disregarded
        '''

        s = np.inf # for later computation

        # set of theta that could be disregarded
        remove_index_set = list(set(old_coefficient.keys()) & set(new_coefficient.keys()))
        
        for theta in remove_index_set:
            
            if new_coefficient[theta] < 0:
                # a quota to determine which support should be disregarded
                # sequential unrestricted minimizations and support reductions
                b = old_coefficient[theta] / (old_coefficient[theta] - new_coefficient[theta])
                # ???
                
                if b < s:
                    s = b
                    remove_support = theta
        
        if s < np.inf:
            return remove_support
        else:
            # this term could be useless
            return False

    def _log_likelihood(self, coefficient):
        '''
        for MLE
        log likelihood, the loss function
        '''
        log_likelihood = 0
        
        for zi in self.Zn:
            l = 0
            
            for thetai in coefficient.keys():
                l += coefficient[thetai] * self._g(thetai, zi)

            l = np.log(l)
            log_likelihood += l
        
        return - 1 * log_likelihood / self.n

    def train(self, initialize=None):
        '''
        training procedure
        initialize should be an integer: the number of initialized supports
        '''
        # initialize support and coefficient
        if initialize is None:
            theta_0 = np.random.choice(self.Zn)
            self.support = {theta_0}
            self.coefficient = {theta_0: 1}
        else:
            theta = np.random.choice(self.Zn, initialize)
            self.support = set(theta)
            self.coefficient = {support: 1 / initialize for support in self.support}
        
        while True:
            # print the objective funtion or loss function
            print(self._log_likelihood(self.coefficient))

            # support reduction, sign is for ending the loop
            sign, self.coefficient, self.support = self._support_reduction(self.coefficient)
            
            if sign == 'Done':
                # coefficient of the new support is less than 0
                # break and return the current coefficient
            	print('coefficient of the new support is less than 0')
            	return self.coefficient
            
            elif sign: 
                # no new support
                # return current coefficient
            	return self.coefficient
            
            else: 
                # new support exists, loop continues
            	pass
    
    def _support_reduction(self, coefficient):
        '''
        support reduction algorithm
        '''
        old_coefficient = coefficient.copy()
        # old support sets
        support = set(old_coefficient.keys())
        new_support = self._new_support(old_coefficient)
        
        if not new_support:
            # no new support, algorithm ended
            return True, self.coefficient, self.support
        
        # add a support and solve the linear equation
        support.add(new_support)
        new_coefficient, sign = self._solve(support)

        if new_coefficient[new_support] < 0:
            # coefficient of the new support is less than 0
            # algorithm ended
            return 'Done', old_coefficient, set(old_coefficient.keys())
            
        while not sign:
            # sequential unrestricted minimizations and support reductions
            '''
            if new_coefficient[new_support] < 0:
                # coefficient of the new support is less than 0
                # algorithm ended
                return 'Done', old_coefficient, set(old_coefficient.keys())
            '''
            # remove support and back to the start of the loop until all coefficients are greater than zero
            remove_support = self._remove_index(old_coefficient, new_coefficient)
            # old_coefficient = new_coefficient.copy()
            if remove_support:
                # find f_j in the next iteration
                lambda_hat = old_coefficient[remove_support] / (old_coefficient[remove_support] - new_coefficient[remove_support])
                f_j = dict()

                for theta in list(set(old_coefficient.keys()) | set(new_coefficient.keys())):
                    if theta in set(old_coefficient.keys()) & set(new_coefficient.keys()):
                        f_j[theta] = old_coefficient[theta] + lambda_hat * (new_coefficient[theta] - old_coefficient[theta])
                    elif theta in set(old_coefficient.keys()) - set(new_coefficient.keys()):
                        f_j[theta] = (1 - lambda_hat) * old_coefficient[theta]
                    elif theta in set(new_coefficient.keys()) - set(old_coefficient.keys()):
                        f_j[theta] = lambda_hat * new_coefficient[theta]

                old_coefficient = f_j.copy()

                support.remove(remove_support)
                new_coefficient, sign = self._solve(support)
        
        return False, new_coefficient, support
    
    def estimator(self, x):
        '''
        a sequence of x is given and output is a sequence of estimate of F(x)
        '''
        X = np.array(x)
        # F(x) estimate
        f = np.zeros(X.shape)
        
        for i in range(len(X)):
            S = 0
            
            for theta in self.coefficient.keys():
                
                #S += self.coefficient[theta] * self._F(theta, X[i])
                S += self.coefficient[theta] * self._g(theta, X[i])
            
            f[i] = S
        
        return f

