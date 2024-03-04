import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

class Perceptron(): 
	def __init__(self, aprendizaje=0.1, n_iter=50):
		self.aprendizaje = aprendizaje
		self.n_iter = n_iter

	def fit(self, X, y):      
		self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])] 
		self.errors_ = []   

		for _ in range(self.n_iter):
			errors = 0
			for xi, label in zip(X, y):
				update = self.aprendizaje * (label-self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)

	def test(self, X, y):      
		self.errors_ = []   
		for _ in range(self.n_iter):
			errors = 0
			for xi, label in zip(X, y):
				update = self.aprendizaje * (label-self.predict(xi))
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self