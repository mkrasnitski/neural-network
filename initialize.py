import pickle
import random
# nodes = [3, 2, 3, 2]
nodes = [784, 16, 16, 10]
# nodes = [4, 4, 4, 4, 4, 4, 4]
# nodes = [10, 10, 10, 10, 10, 10]
activations = [[0]*n for n in nodes]
biases = [None] + [[random.uniform(-10, 10) for i in range(n)] for n in nodes[1:]]
weights = [[[random.uniform(-10, 10) for j in range(nodes[i+1])] for k in range(n)] for i, n in enumerate(nodes[:-1])]
with open('network', 'wb') as f:
	pickle.dump(nodes, f)
	pickle.dump(weights, f)
	pickle.dump(biases, f)