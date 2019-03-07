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
		# self.clear()

	def clear(self):
		self.gradW = [np.zeros(w.shape) for w in self.weights]
		self.gradB = [np.zeros(b.shape) for b in self.biases]

	def save(self):
		with open(self.path, 'wb') as f:
			pickle.dump(self.nodes, f)
			pickle.dump(self.weights, f)
			pickle.dump(self.biases, f)

	def print_last(self):
		for i, (a, y) in enumerate(zip(self.activations[-1], self.y)):
			print(f'{i} - {a:.7f} {y}')

	def get_guess(self):
		return self.activations[-1].argsort()[-3:][::-1]

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

	def gradAW(self, A, W, q):
		if A == W + 1:
			g = np.zeros((self.nodes[W], self.nodes[A]))
			g[:,q] = self.sigmoid_derivs[A][q]*self.activations[W]
			return g
		elif A == W + 2:
			w = self.sigmoid_derivs[A-1][q]*self.sigmoid_derivs[A]*self.weights[A-1][q]
			return np.outer(self.activations[W], w)
		else:
			WG = self.gradAW(A-1, W, q)
			sig = self.sigmoid_derivs[A]
			return sig*np.matmul(WG, self.weights[A-1])

	def gradAB(self, A, B):
		if A - B == 1:
			return np.diag(self.sigmoid_derivs[A])
		elif A - B == 2:
			return self.sigmoid_derivs[A-1][:,None]*self.sigmoid_derivs[A]*self.weights[A-1]
		else:
			return self.sigmoid_derivs[A]*np.matmul(self.gradAB(A-1, B), self.weights[A-1])

	def single_descent(self):
		gradW = [np.empty(w.shape) for w in self.weights]
		gradB = [np.empty(b.shape) for b in self.biases]
		C = 2*(self.activations[-1] - self.y)
		L = len(self.nodes) - 1
		for n in range(L):
			W = np.empty((self.nodes[n+1], self.nodes[n], self.nodes[-1]))
			for q in range(self.nodes[n+1]):
				W[q] = self.gradAW(L, n, q)
			gradW[n] = np.dot(np.swapaxes(W, 0, 1), C)
			gradB[n] = np.dot(self.gradAB(L, n), C)
			# self.gradW[n] = np.dot(np.swapaxes(W, 0, 1), C)
			# self.gradB[n] = np.dot(self.gradAB(L, n), C)
			# print(np.all(self.gradW[1] - gradW[1] < 0.0001))
		return gradW, gradB

	def descend_batch(self, gradW, gradB, size):
		L = len(self.nodes) - 1
		for n in range(L):
			self.weights[n] -= gradW[n]/size
			self.biases[n] -= gradB[n]/size
		# self.clear()
