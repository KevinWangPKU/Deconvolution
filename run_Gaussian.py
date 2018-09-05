import numpy as np
from math import inf, log, exp, sqrt
from scipy.stats import expon, uniform, norm 
from matplotlib import pyplot as plt
from GaussianDeconvolution import Deconvolution

n = 500
mu = expon.rvs(size=n)
X = norm.rvs(loc=mu)

deconv_1 = Deconvolution(n=500, tuning_support=False)

estimator_1 = deconv_1.fit(X, initialize=5)

print('non-tuned:', estimator_1)

# deconv = Deconvolution(n=50, tuning_support=True)

# estimator = deconv.fit(X, initialize=5)

# print('tuned:', estimator)

x = np.linspace(-3, 10, 5000)
f = np.multiply(norm.cdf(x - 1), np.exp(0.5 - x))
plt.plot(x, f, label='ground truth')
# f_hat = deconv.estimator(x, mode='pdf')
# plt.plot(x, f_hat, label='estimation (tuned)')
f_hat_1 = deconv_1.estimator(x, mode='pdf')
plt.plot(x, f_hat_1, label='estimation (not tuned)')

plt.legend()
plt.show()