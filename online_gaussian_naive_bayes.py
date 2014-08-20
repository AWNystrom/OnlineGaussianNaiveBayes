#Written by Andrew Nystrom, AWNystrom@gmail.com, 2014-08-20

from collections import defaultdict
from math import log, e, pi
import numpy as np

class OnlineGaussianNaiveBayes(object):
	def __init__(self, D, priors=False):
		"""
		Online Gaussian Naive Bayes Classifier (OnlineGaussianNaiveBayes)
		
		This class implements an online version of Gaussian Naive Bayes. It
		does this by using an incremental online method for calculating
		variances for each class, requiring each instance to only be seen
		once.
		
		For the online calculation of variance, see
		http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm
		
		Parameters
		----------
		D : dimensionality of the input data.
		
		priors : Whether to use prior probabilities of classes in inference.
		
		Example
		--------
		>>> from online_gaussian_naive_bayes import OnlineGaussianNaiveBayes
		>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = OnlineGaussianNaiveBayes(2)
    >>> for i in xrange(len(X)):
    >>>   clf.fit(X[i], Y[i])
    >>> print(clf.predict([[-0.8, -1]]))
    1
		"""
		
		
		self.D_ = np.int32(D)
		assert self.D_ > 0
		self.priors_ = priors
		self.n_ = defaultdict(lambda: np.int64(0))
		self.mean_ = defaultdict(lambda: np.ndarray(D, dtype=np.float, 
																								buffer=np.zeros(D)))
		self.M2_ = defaultdict(lambda: np.ndarray(D, dtype=np.float, 
																							buffer=np.zeros(D)))
		self.var_ = defaultdict(lambda: np.ndarray(D, dtype=np.float, 
																							buffer=np.zeros(D)))
		self.total_ = np.int64(0)
		self.min_var_ = np.inf
	
	def fit(self, x, c):
		"""
		Shift the mean and variance for class c according to point x.
		
		Parameters
		----------
		x : a vector representing the instance to be trained on.
		
		c : any hashable representation of a class label
		"""
		
		assert len(x) == self.D_
		self.total_ += 1
		self.n_[c] += 1
		n = self.n_[c]
		for i in xrange(self.D_):
			delta = x[i] - self.mean_[c][i]
			self.mean_[c][i] += delta / n
			self.M2_[c][i] = delta*(x[i] - self.mean_[c][i])
			
			if n < 2:
				self.var_[c][i] = np.float(0)
			else:
				self.var_[c][i] = self.M2_[c][i] / (n - 1)
				
			if self.var_[c][i] < self.min_var_ and self.var_[c][i] != 0:
				self.min_var_ = self.var_[c][i]
	
	def _pdf(self, x, m, v):
		
		#Don't divide by zero, but don't use too much smoothing. 100th of
		#the least variance seen.
		v += self.min_var_ / 100
		return -(x-m)**2 * log(e) / (2*v) - 0.5*log(2*pi*v)
	
	def predict(self, x):
		"""
		Perform inference according the Gaussian Naive Bayes.
		
		Parameters
		----------
		x : a vector representing the instance to be trained on.
		"""
	
		#Seen one or no points? Don't even try (no variances calculated).
		if self.n_ < 2:
			return None
			
		best_score_class_pair = (-np.inf, None)
		for c in self.mean_:
			score = sum((self._pdf(x[i], self.mean_[c][i], self.var_[c][i]) \
									 for i in xrange(self.D_)))
									 
			if self.priors_:
				score += log(self.n_[c]) - log(self.total_)
			best_score_class_pair = max(best_score_class_pair, (score, c), 
																	key=lambda item: item[0])
		
		return best_score_class_pair[1]