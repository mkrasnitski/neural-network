import argparse
import math
import pickle
# import matplotlib.pyplot as plt
import numpy as np
from network import Network
np.set_printoptions(suppress=True)

n = Network('network')
image = []
labels = [0]*10
# print(f'{(batch*batch_size + b):03d}', end = ' ', flush=True)
with open(f'image_data/data/000', 'rb') as p:
	image = pickle.load(p)
	label = pickle.load(p)
	labels[label] = 1
for i in range(num):
	x = n.run(np.array(image)/255, labels)
	# cost += x
	print(f'{i} - {x:.15f}')
	n.single_descent()
	n.descend_batch(1)
print()
n.print_last()


# batch_size = 200
# num_batches = 10
# epochs = 20

# images = [None]*batch_size*num_batches
# labels = [None]*batch_size*num_batches
# for i in range(batch_size*num_batches):
# 	with open(f'image_data/data/{i:05d}', 'rb') as f:
# 		images[i] = (pickle.load(f))
# 		label = pickle.load(f)
# 		labels[i] = [1 if j == label else 0 for j in range(10)]
# print('unpickled')

# for e in range(epochs):
# 	print(f'Epoch {e}')
# 	for batch in range(num_batches):
# 		print(f' - Batch {batch} -', end=' ')
# 		try:
# 			cost = 0
# 			for b in range(batch_size):
# 				index = batch*batch_size + b
# 				if b % 10 == 0:
# 					print(f'{index:05d}', end = ' ', flush=True)
# 				x = n.run(images[index], labels[index])
# 				cost += x
# 				n.single_descent()
# 		except KeyboardInterrupt:
# 			n.save()
# 			break
# 		else:
# 			print('-', cost/batch_size)
# 			n.descend_batch(batch_size)
# 	else:
# 		print()
# 		continue
# 	break
# n.save()