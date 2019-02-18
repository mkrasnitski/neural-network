import sys
import math
import pickle
# import matplotlib.pyplot as plt
import numpy as np
from network import Network

num = 10
if len(sys.argv) > 1:
	num = int(sys.argv[1])
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


# batch_size = 100
# num_batches = 1
# epochs = 1
# for e in range(epochs):
# 	print(f'Epoch {e}')
# 	for batch in range(num_batches):
# 		print(f' - Batch {batch} -', end=' ')
# 		try:
# 			cost = 0
# 			for b in range(batch_size):
# 				image = []
# 				labels = [0]*10
# 				if b % 10 == 0:
# 					print(f'{(batch*batch_size + b):03d}', end = ' ', flush=True)
# 				with open(f'image_data/data/{(batch*batch_size + b):03d}', 'rb') as p:
# 					image = pickle.load(p)
# 					label = pickle.load(p)
# 					labels[label] = 1
# 				x = n.run(image, labels)
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