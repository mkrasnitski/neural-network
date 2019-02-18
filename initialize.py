import pickle
import random
import numpy as np
# nodes = [3, 2, 3, 2]
nodes = [784, 16, 16, 10]
# nodes = [4, 4, 4, 4, 4, 4, 4]
# nodes = [10, 10, 10, 10]
lim = 10
activations = [np.zeros((n,)) for n in nodes]
biases = [lim*(np.random.rand(n)*2 - 1) for n in nodes[1:]]
weights = [lim*(np.random.rand(n, nodes[i+1])*2 - 1) for i, n in enumerate(nodes[:-1])]
with open('network', 'wb') as f:
	pickle.dump(nodes, f)
	pickle.dump(weights, f)
	pickle.dump(biases, f)