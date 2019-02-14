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
				for j in range(self.nodes[A_layer-1]):
					g[i] += W[j]*self.sigmoid_derivs[A_layer][i]*self.weights[A_layer-1][j][i]
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
				for j in range(self.nodes[A_layer-1]):
					g[i] += B[j]*self.sigmoid_derivs[A_layer][i]*self.weights[A_layer-1][j][i]
			return g

	def weight_gradient(self, W_layer, p, q):
		L = len(self.nodes) - 1
		W = self.gradAW(L, W_layer, p, q)
		C = self.gradCA()
		return self.dot(C, W)

	def bias_gradient(self, B_layer, p):
		L = len(self.nodes) - 1
		B = self.gradAB(L, B_layer, p)
		C = self.gradCA()
		return self.dot(C, B)
		
	def gradAW(self, A_layer, W_layer, i, p, q):
		if A_layer == W_layer + 1:
			if i == q:
				return self.sigmoid(self.z[A_layer][i], True) * self.activations[W_layer][p]
			else:
				return 0
		elif A_layer == W_layer + 2:
			return self.gradAA(A_layer, i, q)*self.gradAW(A_layer-1, W_layer, q, p, q)
		else:
			return sum(self.gradAA(A_layer, i, j)*self.gradAW(A_layer-1, W_layer, j, p, q) for j in range(self.nodes[A_layer-1]))

	def gradAB(self, A_layer, B_layer, i, p):
		if B_layer == 0: raise Exception("there are no biases on level 0")
		if A_layer == B_layer:
			if i == p:
				return self.sigmoid(self.z[A_layer][i], True)
			else:
				return 0
		elif A_layer == B_layer + 1:
			return self.gradAA(A_layer, i, p)*self.gradAB(A_layer-1, B_layer, p, p)
		else:
			return sum(self.gradAA(A_layer, i, j)*self.gradAB(A_layer-1, B_layer, j, p) for j in range(self.nodes[A_layer-1]))

	def weight_gradient(self, W_layer, p, q):
		L = len(self.nodes) - 1
		W = []
		for i in range(self.nodes[L]):
			W.append(self.gradAW(L, W_layer, i, p, q))
		C = self.gradCA()
		return sum(c*w for c, w in zip(C, W))

	def bias_gradient(self, B_layer, p):
		L = len(self.nodes) - 1
		B = []
		for i in range(self.nodes[L]):
			B.append(self.gradAB(L, B_layer, i, p))
		C = self.gradCA()
		return sum(c*b for c, b in zip(C, B))