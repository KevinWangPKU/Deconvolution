import numpy as np
from math import inf, log, exp, sqrt
from scipy.stats import expon, uniform
from matplotlib import pyplot as plt
from Deconvolution import Deconvolution

def P(x):
    return x + 0.5 * (x ** 2)

def K(x):
    if x < 0:
        return 0
    else:
        return 1 - exp(- 1 * x)

n = 1000
noise = expon.rvs(size=n)
F = uniform.rvs(size=n)
X = 5 * (F ** 2)
Z = X + noise

deconv = Deconvolution(Z, method='least_square', P=P)

estimator = deconv.train(initialize=None)


x = np.linspace(0, 5, 1000)
f = np.sqrt(x / 5)
plt.plot(x, f, label='ground truth')
f_hat = deconv.estimator(x)
plt.plot(x, f_hat, label='estimation')
plt.legend()
plt.show()