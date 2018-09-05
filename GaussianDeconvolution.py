import numpy as np
from math import inf, log, exp, sqrt, pi
from scipy.stats import expon, uniform, norm
from matplotlib import pyplot as plt

class Deconvolution:
	def __init__(self, n=500, tuning_support=True):
		'''
		divide the interval [x_(1), x_(n)] into n small intervals with the same length
		tuning_support: if true, tune the support points
		'''
		self.n_split = n
		self.tuning_support = tuning_support

	def _g(self, theta, x):
		'''
		probability density function of N(x, 1)
		'''
		return norm.pdf(x, loc=theta)

	def _c1(self, theta, coefficient=None):
		'''
		c_1(theta, g_bar), for computing new support
		'''
		c1 = 0

		for zi in self.Zn:
			c1 += self._g(theta, zi) / self._g_bar(zi)

		return 1 - c1 / self.n

	def _c2(self, theta):
		'''
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

	def _F(self, theta, x):
		'''
		cumulative density function of N(theta, 1)
		'''
		return norm.cdf(x, loc=theta)

	def _new_support(self, coefficient):
		'''
		find new support
		'''
		support = set(coefficient.keys())
		# set of new supports
		new_support_set = list(set(self.support_set) - support)
		
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

		
		# A(alpha_1, ..., alpha_m) = b
		A = np.zeros((m, m))
		b = np.zeros(m)
		
		# obtian A and b
		for i in range(m):
			for j in range(m):
				a = 0
				for zi in self.Zn:
					a += self._g(support[i], zi) * self._g(support[j], zi) / (self._g_bar(zi) ** 2)
				a = a / self.n

				A[i, j] = a
			b[i] = 1 - 2 * self._c1(support[i])
			
		# coefficients: (alpha_1, ..., alpha_m)
		coef = np.linalg.solve(A, b)
		coefficient = dict()
		
		# sign: denotes whether there is a negative term in coefficients
		# sign: negative coefficient exists, False; otherwise, True
		sign = True
		
		for i in range(m):
			coefficient[support[i]] = coef[i]
			if coef[i] < 0:
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

	def fit(self, Z, initialize=None, learning_rate=0.1):
		'''
		training procedure
		initialize should be an integer: the number of initialized supports
		learning rate: for tuning support; if self.tuning_support == False, learning rate is useless
		'''
		# initialize support and coefficient
		self.Zn = Z
		self.n = len(Z)
		self.support_set = list(np.linspace(np.min(self.Zn), np.max(self.Zn), num=self.n_split, endpoint=True))
		self.learning_rate = learning_rate

		if initialize is None:
			theta_0 = np.random.choice(self.support_set)
			self.support = {theta_0}
			self.coefficient = {theta_0: 1}
		else:
			theta = np.random.choice(self.support_set, initialize)
			self.support = set(theta)
			self.coefficient = {support: 1 / initialize for support in self.support}

		while True:
			# print the objective funtion or loss function
			print('log likelihood:', self._log_likelihood(self.coefficient))
			
			# support reduction, sign is for ending the loop
			sign, self.coefficient, self.support = self._support_reduction(self.coefficient)
			
			if sign:
				if self.tuning_support == True:
					tuning = self._tuning()
					while tuning:
						tuned_coefficient = self._tuning_support(self.learning_rate)
						self.support_set.extend(list(tuned_coefficient.keys()))
						sign, self.coefficient, self.support = self._support_reduction(tuned_coefficient)
						
						print('log likelihood:', self._log_likelihood(self.coefficient))

						tuning = self._tuning()
				else:
					pass
					
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
			print('coefficient of the new support is less than 0')
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

		# to make sure of the monotonicity
		lambda_choice = [1, 0.75, 0.5, 0.25]
		for lambda0 in lambda_choice:
			next_coefficient = self._linear_combination(coefficient, new_coefficient, lambda0)
			if self._log_likelihood(next_coefficient) < self._log_likelihood(coefficient):
				new_coefficient = next_coefficient.copy()
				break
		
		return False, new_coefficient, support
    
	def estimator(self, x, mode='cdf', coefficient=None):
		'''
		a sequence of x is given and output is a sequence of estimate of F(x)
		'''
		if coefficient is None:
			coefficient = self.coefficient.copy()

		if isinstance(x, float) or isinstance(x, int):
			X = np.array([x])
		else:
			X = np.array(x)
		# F(x) estimate
		f = np.zeros(X.shape)
		for i in range(len(X)):
			S = 0
			for theta in coefficient.keys():
				if mode == 'cdf':
					S += coefficient[theta] * self._F(theta, X[i])
				elif mode == 'pdf':
					S += coefficient[theta] * self._g(theta, X[i])
				elif mode == 'cdf_of_theta':
					S += coefficient[theta] * (theta < X[i])

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
		
		return new_coefficient

	def _tuning(self):
		'''
		decide whether a tuning support process is needed
		'''
		if np.sum(np.abs(self._tau_derivative(self.coefficient)) < 0.01) == len(self.coefficient):
			# all components of the vector _tau_derivative are neglectable
			return False
		else:
			return True

	def _add(self, epsilon):
		'''
		{f_{theta_i - epsilon * h_i: alpha_i}}
		h = derivative(tau(0))
		'''
		coefficient_epsilon = dict()
		h = self._tau_derivative(self.coefficient) #derivative(tau(0))
		support_set = list(self.coefficient.keys())

		for i in range(len(support_set)):
			theta = support_set[i]
			coefficient_epsilon[theta - epsilon * h[i]] = self.coefficient[theta]
			# slightly tune the support grid, while keep the coef fixed

		# print('add epsilon:', coefficient_epsilon)

		return coefficient_epsilon


	def _tau_derivative(self, coefficient):
		'''
		derivative(tau(0))
		'''

		h = np.zeros(len(coefficient))
		support_set = list(coefficient.keys())
		
		for i in range(len(h)):
			hi = 0

			for zj in self.Zn:
				# according to the derivative of theta of f_theta...
				hi += self._g(support_set[i], zj) * (zj - support_set[i]) / float(self.estimator(zj, mode='pdf', coefficient=coefficient))
				
			
			h[i] = - coefficient[support_set[i]] * hi

		# print('tay derivative:', h)
		return h / self.n

	def _mu(self, epsilon):
		'''
		mu_{epsilon}(h)
		'''
		mu = self._log_likelihood(self._add(epsilon)) - self._log_likelihood(self.coefficient)
		# print('mu:', mu)

		return mu


	def _mu_derivative(self, epsilon):
		'''
		mu'(h) used for computing learning rate
		'''

		h = - 1 * self._tau_derivative(self.coefficient)
		tau_epsilon_h = self._tau_derivative(self._add(epsilon))
		mu_derivative = np.sum(np.multiply(h, tau_epsilon_h))
		# print('mu derivative:', mu_derivative)

		return mu_derivative


	def _tuning_support(self, learning_rate=0.1):
		epsilon_0 = learning_rate

		while True:
			epsilon_l = 0
			epsilon_u = epsilon_0

			if self._mu(epsilon_u) < 0:
				epsilon = epsilon_u
				# print('learning rate:', epsilon)
				return self._add(epsilon)
			
			else:
				sign = True
				
				while sign:
					epsilon_n = (epsilon_l * self._mu_derivative(epsilon_u) - epsilon_u * self._mu_derivative(epsilon_l)) / (self._mu_derivative(epsilon_u) - self._mu_derivative(epsilon_l))
					
					if self._mu_derivative(epsilon_n) > 0:
						epsilon_u = epsilon_n
					
					else:
						epsilon_l = epsilon_n

					if abs(self._mu_derivative(epsilon_n)) < 0.5:
						sign = False
					
					else:
						pass
				
				if self._mu(epsilon_u) > 0:
					c = 0.1
					epsilon_0 = c * epsilon_n 
					continue










