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

	def save(self):
		with open(self.path, 'wb') as f:
			pickle.dump(self.nodes, f)
			pickle.dump(self.weights, f)
			pickle.dump(self.biases, f)

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
			self.z[l] = self.biases[l-1] + np.dot(self.weights[l-1].T, self.activations[l-1])
			self.activations[l] = self.sigmoid(self.z[l], False)
			self.sigmoid_derivs[l] = self.sigmoid(self.z[l], True)

		cost = 0
		for a, y in zip(self.activations[-1], self.y):
			cost += (a-y)**2
		return cost

	def gradAWB(self, A_layer, WB_layer, q, t):
		if A_layer == WB_layer + 1:
			if t == 'b':
				g = np.zeros(self.nodes[A_layer])
				g[q] = self.sigmoid_derivs[A_layer][q]
				return g
			else:
				g = np.zeros((self.nodes[WB_layer], self.nodes[A_layer]))
				g[:,q] = self.sigmoid_derivs[A_layer][q]*self.activations[WB_layer]
				return g
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
		gradW = [np.empty(w.shape) for w in self.weights]
		gradB = [np.empty(b.shape) for b in self.biases]
		C = 2*(self.activations[-1] - self.y)
		L = len(self.nodes) - 1
		for n in range(L):
			WG = np.empty((self.nodes[n+1], self.nodes[n], self.nodes[-1]))
			BG = np.empty((self.nodes[n+1], self.nodes[-1]))
			for q in range(self.nodes[n+1]):
				WG[q] = self.gradAWB(L, n, q, 'w')
				BG[q] = self.gradAWB(L, n, q, 'b')
			gradW[n] = np.dot(np.transpose(WG, (1, 0, 2)), C)
			gradB[n] = np.dot(BG, C)
		return gradW, gradB

	def descend_batch(self, gradW, gradB, size):
		L = len(self.nodes) - 1
		for n in range(L):
			self.weights[n] -= gradW[n]/size
			self.biases[n] -= gradB[n]/size
