import pickle
import math
import numpy as np

class Network:
	def __init__(self, path):
		self.path = path
		with open(path, 'rb') as f:
			self.nodes = pickle.load(f)
			self.weights = pickle.load(f)
			self.biases = pickle.load(f)
		self.activations = [np.empty((n,)) for n in self.nodes]
		self.z = [np.empty((n,)) for n in self.nodes]
		self.sigmoid_derivs = [np.empty((n,)) for n in self.nodes]
		self.clear()

		self.size = len(self.nodes)

	def save(self):
		with open(self.path, 'wb') as f:
			pickle.dump(self.nodes, f)
			pickle.dump(self.weights, f)
			pickle.dump(self.biases, f)

	def clear(self):
		self.gradW = [np.zeros(w.shape) for w in self.weights]
		self.gradB = [np.zeros(b.shape) for b in self.biases]

	def print_last(self):
		for a, y in zip(self.activations[-1], self.y):
			print(a, y)

	def sigmoid(self, x, deriv):
		if not deriv:
			return (x/(1 + abs(x))+1)/2
		return 1/(2*(1+abs(x))**2)

	# def sigmoid(self, x, deriv):
	# 	if not deriv:
	# 		return 1/(1+math.exp(-x))
	# 	return math.exp(-x)/((1+math.exp(-x))**2)

	def run(self, initial, output):
		assert(len(initial) == self.nodes[0])
		assert(len(output) == self.nodes[-1])
		self.activations[0] = initial
		self.y = output

		for l in range(1, self.size):
			for i in range(self.nodes[l]):
				z = np.dot(self.activations[l-1], self.weights[l-1][:,i])
				self.z[l][i] = z + self.biases[l-1][i]
				self.activations[l][i] = self.sigmoid(self.z[l][i], False)
				self.sigmoid_derivs[l][i] = self.sigmoid(self.z[l][i], True)
		cost = 0
		for a, y in zip(self.activations[-1], self.y):
			cost += (a-y)**2
		return cost

	def gradCA(self):
		C = [None]*(self.nodes[-1])
		for i in range(self.nodes[-1]):
			C[i] = 2*(self.activations[-1][i] - self.y[i])
		return C
		
	def gradAWB(self, A_layer, WB_layer, p, q, t):
		if A_layer == WB_layer + 1:
			g = np.zeros(self.nodes[A_layer])
			g[q] = self.sigmoid_derivs[A_layer][q]
			if t == 'w':
				g[q] *= self.activations[WB_layer][p]
			return g
		elif A_layer == WB_layer + 2:
			wb = self.sigmoid_derivs[A_layer-1][q]
			if t == 'w':
				wb *= self.activations[WB_layer][p]
			return wb*np.multiply(self.sigmoid_derivs[A_layer], self.weights[A_layer-1][q])
		else:
			WB = self.gradAWB(A_layer-1, WB_layer, p, q, t)
			sig = self.sigmoid_derivs[A_layer]
			dot = np.dot(WB, self.weights[A_layer-1])
			return sig*dot

	def single_descent(self):
		C = self.gradCA()
		L = len(self.nodes) - 1
		for n in range(L):
			for q in range(self.nodes[n+1]):
				for p in range(self.nodes[n]):
					# WG = self.gradAW(L, n, p, q)
					WG = self.gradAWB(L, n, p, q, 'w')
					self.gradW[n][p][q] += np.dot(C, WG)
				# BG = self.gradAB(L, n, q)
				BG = self.gradAWB(L, n, None, q, 'b')
				self.gradB[n][q] += np.dot(C, BG)

	def descend_batch(self, size):
		C = self.gradCA()
		L = len(self.nodes) - 1
		for n in range(L):
			self.weights[n] -= self.gradW[n]/size
			self.biases[n] -= self.gradB[n]/size
		self.clear()