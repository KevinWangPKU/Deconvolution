#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:04:50 2018

@author: WangJianqiao
"""
import numpy as np
from math import inf, log, exp, sqrt
from scipy.stats import expon, uniform
from matplotlib import pyplot as plt

class Deconvolution:
    def __init__(self, method='least_square', P=None, K=None):
        '''
        method: 'least_square' or 'mle'
        P: for 'least_square'; primitive of p(x)
        K: for 'mle'; primitive of k(x)
        n: sample size
        '''
        self.P = P
        self.K = K
        self.n = len(Zn)
        self.method = method
    
    def _inner_product1(self, theta_1, theta_2):
        '''
        for least square
        inner product between s_theta_1, s_theta_2, <s_theta1, s_theta2>
        '''
        x = min(theta_1, theta_2)
        
        return x - (0.5 * (x ** 2) / theta_1) - (0.5 * (x ** 2) / theta_2) + ((x ** 3) / (3 * theta_1 * theta_2))
    
    def _indicator(self, x, theta):
        '''
        for least square
        indicator function
        '''
        if x > theta:
            return 1
        else:
            return 0
        
    def _inner_product2(self, theta):
        '''
        for least square
        innner product beween s_theta and dU_n: <s_theta, dUn>
        '''
        s = 0
        
        for zi in self.Zn:
            s += (self.P(theta - zi) - self.P(0)) * self._indicator(theta, zi)
        
        return (theta / 2) - (s / (self.n * theta))
    
    def _s(self, theta, x):
        '''
        for least square
        s_theta(x)
        '''
        if x < theta:
            return 1 - x / theta
        
        else:
            return 0

    def _g(self, theta, x):
        '''
        for MLE
        g_theta(x)
        '''
        return (self.K(x) - self.K(x-theta)) / theta

    def _F(self, theta, x):
        '''
        for MLE
        F_theta(x)
        '''
        if x > theta:
            return 1
        else:
            return x / theta

    def _c1(self, theta, coefficient=None):
        '''
        least square: c_1(theta, s), directional derivative function 
                      of objective function Q
        mle: c_1(theta, g_bar), for computing new support
        '''
        if self.method == 'least_square':
            
            a = self._inner_product2(theta)
            s = 0
            
            for theta_i in coefficient.keys():
                s += coefficient[theta_i] * self._inner_product1(theta, theta_i)
            
            return s - a
        
        elif self.method == 'mle':
            
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
            c2 += (self._g(theta, zi) ** 2) / (self._g_bar(zi) ** 2)

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
        
        if self.method == 'least_square':
            # find the least derivative
            c = 0
            
            for i in range(len(new_support_set)):
                
                c1 = self._c1(new_support_set[i], coefficient)
                
                if c1 < c:
                    c = c1
                    new_support = new_support_set[i]
            
            if c < 0:
                return new_support
            
            else:
                # all derivatives are greater than zero, algorithm ended
                return False
        
        elif self.method == 'mle':
            
            if len(support) == self.n:
                # no new support, algorithm ended
                return False
            
            # find the least c_1(theta, g_bar) / sqrt(c_2(theta))
            support_eval = np.zeros(len(new_support_set))
            
            for i in range(len(new_support_set)):
                c1 = self._c1(new_support_set[i])
                c2 = self._c2(new_support_set[i])
                support_eval[i] = c1 / sqrt(c2)

            return new_support_set[np.argmin(support_eval)]
    

    def _solve(self, support_set):
        '''
        solve liearn equation to obtain new coefficients
        '''
        # number of supports
        m = len(support_set)
        # support set
        support = list(support_set)
        # coefficient: {theta: alpha}
        coefficient = dict()
        
        if self.method == 'least_square':
            '''
            A(alpha_2, ..., alpha_m)' = b
            alpha_1 = 1 - sum(alpha_i)
            '''
            A = np.mat(np.zeros((m - 1, m - 1)))
            b = np.mat(np.zeros((m - 1, 1)))
            theta1 = support[0] # for alpha_1
            
            # obtian A and b
            for i in range(m - 1):
                for j in range(m - 1):
                    A[i, j] = self._inner_product1(theta1, theta1) + self._inner_product1(support[i + 1], support[j + 1]) - self._inner_product1(support[i + 1], theta1) - self._inner_product1(support[j + 1], theta1)
                b[i, 0] = self._inner_product1(theta1, theta1) - self._inner_product1(support[i + 1], theta1) - self._inner_product2(theta1) + self._inner_product2(support[i + 1])
            
            # coefficients
            # coef: (alpha_2, ... , alpha_m)
            coef = np.matmul(A.I, b)
            alpha1 = 1 - coef.sum()
            
            # sign: denotes whether there is a negative term in coefficients
            # sign: negative coefficient exists, False; otherwise, True
            sign = True
            for i in range(m):
                
                if i == 0:
                    coefficient[theta1] = alpha1
                    if alpha1 < 0:
                        sign = False
                
                else:
                    coefficient[support[i]] = coef[i - 1, 0]
                    if coef[i - 1, 0] < 0:
                        sign = False
        
        elif self.method == 'mle':
            '''
            A(alpha_1, ..., alpha_m) = b
            '''
            A = np.mat(np.zeros((m, m)))
            b = np.mat(np.zeros((m, 1)))
            
            # obtian A and b
            for i in range(m):
                for j in range(m):
                    a = 0
                    for zi in self.Zn:
                        a += self._g(support[i], zi) * self._g(support[j], zi) / (self._g_bar(zi) ** 2)
                    a = a / self.n

                    A[i, j] = a
                b[i, 0] = 1 - 2 * self._c1(support[i])
            
            # coefficients: (alpha_1, ..., alpha_m)
            coef = np.matmul(A.I, b)
            coefficient = dict()
            
            # sign: denotes whether there is a negative term in coefficients
            # sign: negative coefficient exists, False; otherwise, True
            sign = True
            
            for i in range(m):
                coefficient[support[i]] = coef[i, 0]
                if coef[i, 0] < 0:
                    sign = False
        
        return coefficient, sign
    
    def _remove_index(self, old_coefficient, new_coefficient):
        '''
        support needed to be disregarded
        '''

        s = inf # for later computation

        # set of theta that could be disregarded
        remove_index_set = list(set(old_coefficient.keys()) & set(new_coefficient.keys()))
        
        for theta in remove_index_set:
            
            if new_coefficient[theta] < 0:
                # a quota to determine which support should be disregarded
                # sequential unrestricted minimizations and support reductions
                b = old_coefficient[theta] / (old_coefficient[theta] - new_coefficient[theta])
                
                if b < s:
                    s = b
                    remove_support = theta
        
        if s < inf:
            return remove_support
        else:
            # this term could be useless
            return False
    
    def _Q(self, coefficient):
        '''
        for least square
        objective function
        '''
        Q = 0
        
        for thetai in coefficient.keys():
            for thetaj in coefficient.keys():
                Q += 0.5 * coefficient[thetai] * coefficient[thetaj] * self._inner_product1(thetai, thetaj)
            Q -= coefficient[thetai] * self._inner_product2(thetai)
        
        return Q

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

            l = log(l)
            log_likelihood += l
        
        return - 1 * log_likelihood / self.n

    def fit(self, Zn, initialize=None):
        '''
        training procedure
        initialize should be an integer: the number of initialized supports
        '''
        # initialize support and coefficient

        # Zn: observed data
        self.Zn = Zn

        if initialize is None:
            theta_0 = np.random.choice(self.Zn)
            self.support = {theta_0}
            self.coefficient = {theta_0: 1}
        else:
            theta = np.random.choice(self.Zn, initialize)
            self.support = set(theta)
            self.coefficient = {support: 1 / initialize for support in self.support}
            if self.method == 'least_square':
                # sequential unrestricted minimizations and support reductions
                # to ensure Q(coefficient) < Q(0)
                while self._Q(self.coefficient) >= 0:
                    
                    coefficient = self.coefficient.copy()
                    # two variables to ensure the sum of coefficients is one
                    high = 1
                    i = 0
                    
                    for support in self.estimator.keys():
                        
                        if i == initialize - 1:
                            coefficient[support] = high
                            break
                        
                        alpha = np.random.uniform(high=high)
                        high -= alpha
                        coefficient[support] = alpha
                        i += 1
                    
                    self.coefficient = coefficient.copy()
            
            elif self.method == 'mle':
                # already satisfy l(coefficient) < l(0)
                pass
        
        while True:
            # print the objective funtion or loss function
            if self.method == 'least_square':
                print(self._Q(self.coefficient))
            elif self.method == 'mle':
                print(self._log_likelihood(self.coefficient))

            # support reduction, sign is for ending the loop
            sign, self.coefficient, self.support = self._support_reduction(self.coefficient)
            
            if sign == 'Done':
                # coefficient of the new support is less than 0
                # break and return the current coefficient
            	# print('coefficient of the new support is less than 0')
            	return
            
            elif sign: 
                # no new support
                # return current coefficient
            	return
            
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
            
            # remove support and back to the start of the loop until all coefficients are greater than zero
            remove_support = self._remove_index(old_coefficient, new_coefficient)
            # old_coefficient = new_coefficient.copy()
            if remove_support:
                # find f_j in the next iteration
                lambda_hat = old_coefficient[remove_support] / (old_coefficient[remove_support] - new_coefficient[remove_support])

                old_coefficient = self._linear_combination(old_coefficient, new_coefficient, lambda_hat)

                support.remove(remove_support)
                new_coefficient, sign = self._solve(support)

        if self.method == 'mle':
            # to make sure of the monotonicity
            lambda_choice = [1, 0.75, 0.5, 0.25]
            for lambda0 in lambda_choice:
                next_coefficient = self._linear_combination(coefficient, new_coefficient, lambda0)
                if self._log_likelihood(next_coefficient) < self._log_likelihood(coefficient):
                    new_coefficient = next_coefficient.copy()
                    break
        else:
            # least square
            pass

        
        return False, new_coefficient, support

    def support(self):
        '''
        return supports and weights
        '''
        return self.coefficient
    
    def estimate(self, x):
        '''
        a sequence of x is given and output is a sequence of estimate of F(x)
        '''
        X = np.array(x)
        # F(x) estimate
        f = np.zeros(X.shape)
        
        for i in range(len(X)):
            S = 0
            
            for theta in self.coefficient.keys():
                
                if self.method == 'least_square':
                    S += self.coefficient[theta] * self._s(theta, X[i])
                
                elif self.method == 'mle':
                    S += self.coefficient[theta] * self._F(theta, X[i])
            
            if self.method == 'least_square':
                f[i] = 1-S
            
            elif self.method == 'mle':
                f[i] = S
        
        return f

    def _linear_combination(self, coefficient1, coefficient2, lambda0):
        '''
        return coefficient1 + lambda0 * (coefficient2 - coefficient1)
        '''
        new_coefficient = dict()

        for theta in list(set(coefficient1.keys()) | set(coefficient2.keys())):
            if theta in set(coefficient1.keys()) & set(coefficient2.keys()):
                coefficient = coefficient1[theta] + lambda0 * (coefficient2[theta] -  coefficient1[theta])
            elif theta in set(coefficient1.keys()) - set(coefficient2.keys()):
                coefficient = (1 - lambda0) * coefficient1[theta]
            elif theta in set(coefficient2.keys()) - set(coefficient1.keys()):
                coefficient = lambda0 * coefficient2[theta]
            else:
                pass

            if coefficient > 0:
                new_coefficient[theta] = coefficient
            else:
                pass

        return new_coefficient


