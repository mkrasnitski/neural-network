import pickle
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
			print(f'{a:.7f}', y)

	def sigmoid(self, x, deriv):
		if not deriv:
			return (x/(1 + abs(x))+1)/2
		return 1/(2*(1+abs(x))**2)

	def run(self, initial, output):
		assert(len(initial) == self.nodes[0])
		assert(len(output) == self.nodes[-1])
		self.activations[0] = initial
		self.y = np.array(output)

		for l in range(1, len(self.nodes)):
			for i in range(self.nodes[l]):
				z = np.dot(self.activations[l-1], self.weights[l-1][:,i])
				self.z[l][i] = z + self.biases[l-1][i]
				self.activations[l][i] = self.sigmoid(self.z[l][i], False)
				self.sigmoid_derivs[l][i] = self.sigmoid(self.z[l][i], True)
		cost = 0
		for a, y in zip(self.activations[-1], self.y):
			cost += (a-y)**2
		return cost

	def gradAWB(self, A_layer, WB_layer, q, t):
		if A_layer == WB_layer + 1:
			g = np.zeros(self.nodes[A_layer])
			g[q] = self.sigmoid_derivs[A_layer][q]
			if t == 'b':
				return g
			return np.outer(self.activations[WB_layer], g)
		elif A_layer == WB_layer + 2:
			wb = self.sigmoid_derivs[A_layer-1][q]*self.weights[A_layer-1][q]*self.sigmoid_derivs[A_layer]
			if t == 'b':
				return wb
			return np.outer(self.activations[WB_layer], wb)
		else:
			WB = self.gradAWB(A_layer-1, WB_layer, q, t)
			sig = self.sigmoid_derivs[A_layer]
			return sig*np.matmul(WB, self.weights[A_layer-1])

	def single_descent(self):
		C = 2*(self.activations[-1] - self.y)
		L = len(self.nodes) - 1
		for n in range(L):
			WG = np.empty((self.nodes[n+1], self.nodes[n], self.nodes[-1]))
			BG = np.empty((self.nodes[n+1], self.nodes[-1]))
			for q in range(self.nodes[n+1]):
				WG[q] = self.gradAWB(L, n, q, 'w')
				BG[q] = self.gradAWB(L, n, q, 'b')
			self.gradW[n] += np.dot(np.transpose(WG, (1, 0, 2)), C)
			self.gradB[n] += np.dot(BG, C)

	def descend_batch(self, size):
		L = len(self.nodes) - 1
		for n in range(L):
			self.weights[n] -= self.gradW[n]/size
			self.biases[n] -= self.gradB[n]/size
		self.clear()