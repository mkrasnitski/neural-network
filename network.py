import pickle
import math
import pp
import sys
# import matplotlib.pyplot as plt

class Network:
	def __init__(self, path):
		with open(path, 'rb') as f:
			self.nodes = pickle.load(f)
			self.weights = pickle.load(f)
			self.biases = pickle.load(f)
		self.activations = [[0]*n for n in self.nodes]
		self.z = [[0]*n for n in self.nodes]
		self.sigmoid_derivs = [[None]*n for n in self.nodes]

		self.gradW = [[[0 for q in p] for p in w] for w in self.weights]
		self.gradB = [None] + [[0 for p in b] for b in self.biases[1:]]

		self.size = len(self.nodes)

	def dot(self, x, y):
		assert(isinstance(x, list))
		assert(isinstance(y, list))
		s = 0
		for a, b in zip(x, y):
			s += a*b
		return s

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
				z = 0
				for j in range(self.nodes[l-1]):
					z += self.activations[l-1][j]*self.weights[l-1][j][i]
				self.z[l][i] = z + self.biases[l][i]
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
		
	def gradAW(self, A_layer, W_layer, p, q):
		if A_layer == W_layer + 1:
			g = [0]*self.nodes[A_layer]
			g[q] = self.sigmoid_derivs[A_layer][q] * self.activations[W_layer][p]
			return g
		elif A_layer == W_layer + 2:
			g = [None]*self.nodes[A_layer]
			w = self.sigmoid_derivs[A_layer-1][q] * self.activations[W_layer][p]
			for i in range(self.nodes[A_layer]):
				a = self.sigmoid_derivs[A_layer][i]*self.weights[A_layer-1][q][i]
				g[i] = a*w
			return g
		else:
			g = [0]*self.nodes[A_layer]
			W = self.gradAW(A_layer-1, W_layer, p, q)
			for i in range(self.nodes[A_layer]):
				sig = self.sigmoid_derivs[A_layer][i]
				for j in range(self.nodes[A_layer-1]):
					g[i] += W[j]*sig*self.weights[A_layer-1][j][i]
			return g

	def gradAB(self, A_layer, B_layer, p):
		if B_layer == 0: raise Exception("there are no biases on level 0")
		if A_layer == B_layer:
			g = [0]*self.nodes[A_layer]
			g[p] = self.sigmoid_derivs[A_layer][p]
			return g
		elif A_layer == B_layer + 1:
			g = [None]*self.nodes[A_layer]
			b = self.sigmoid_derivs[A_layer-1][p]
			for i in range(self.nodes[A_layer]):
				a = self.sigmoid_derivs[A_layer][i]*self.weights[A_layer-1][p][i]
				g[i] = a*b
			return g
		else:
			g = [0]*self.nodes[A_layer]
			B = self.gradAB(A_layer-1, B_layer, p)
			for i in range(self.nodes[A_layer]):
				sig = self.sigmoid_derivs[A_layer][i]
				for j in range(self.nodes[A_layer-1]):
					g[i] += B[j]*sig*self.weights[A_layer-1][j][i]
			return g

	def descend(self):
		C = self.gradCA()
		L = len(self.nodes) - 1
		for n in range(L):
			for q in range(self.nodes[n+1]):
				for p in range(self.nodes[n]):
					WG = self.gradAW(L, n, p, q)
					self.gradW[n][p][q] = self.weights[n][p][q] - self.dot(C, WG)
				BG = self.gradAB(L, n+1, q)
				self.gradB[n+1][q] = self.biases[n+1][q] - self.dot(C, BG)
		self.weights = [[w[:] for w in wg] for wg in self.gradW]			
		self.biases = [bg[:] if bg else None for bg in self.gradB]


num = 100
if len(sys.argv) > 1:
	num = int(sys.argv[1])
n = Network('network')
costs = []
x = math.inf
for i in range(num):
# while x > 0.001:
	x = n.run([i/n.nodes[0] for i in range(n.nodes[0])], [i/n.nodes[-1] for i in range(n.nodes[-1])])
	costs.append(x)
	print(f'{i} - {x:.15f}')
	n.descend()
# plt.plot(range(len(costs)), costs)
# plt.ylim(bottom=0)
# plt.savefig('figure.png')

print()
n.print_last()