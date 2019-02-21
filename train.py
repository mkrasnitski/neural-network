import argparse
import pickle
import numpy as np
from network import Network
np.set_printoptions(suppress=True)

def read_file(path):
	image = label = None
	with open(path, 'rb') as f:
		image = pickle.load(f)
		label = pickle.load(f)
		labels = [1 if i == label else 0 for i in range(10)]
	return image, labels

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
parser.add_argument('-bs', '--batch_size', required=True)
parser.add_argument('-nb', '--num_batches', required=True)
parser.add_argument('-e', '--epochs', required=True)
args = parser.parse_args()

path = args.path
batch_size = int(args.batch_size)
num_batches = int(args.num_batches)
epochs = int(args.epochs)

n = Network(path)

images = [None]*batch_size*num_batches
labels = [None]*batch_size*num_batches
for i in range(batch_size*num_batches):
	images[i], labels[i] = read_file(f'image_data/data/{i:05d}')

for e in range(epochs):
	print(f'Epoch {e}')
	for batch in range(num_batches):
		print(f' - Batch {batch} -', end=' ')
		try:
			cost = 0
			for b in range(batch_size):
				index = batch*batch_size + b
				if b % 100 == 0:
					print(f'{index:05d}', end = ' ', flush=True)
				x = n.run(images[index], labels[index])
				cost += x
				n.single_descent()
		except KeyboardInterrupt:
			n.save()
			break
		else:
			print('-', cost/batch_size)
			n.descend_batch(batch_size)
	else:
		print()
		continue
	break
n.save()