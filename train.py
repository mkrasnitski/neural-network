import argparse
import pickle
import numpy as np
import signal
from multiprocessing import Pool, JoinableQueue
from multiprocessing.managers import SyncManager
from network import Network
np.set_printoptions(suppress=True)

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def read_file(path):
	image = label = None
	with open(path, 'rb') as f:
		image = pickle.load(f)
		label = pickle.load(f)
		labels = [1 if i == label else 0 for i in range(10)]
	return image, labels

def single_example(net, index, image, label):
	c = net.run(image, label)
	if index % 100 == 0:
		print(f'{index:05d}', end=' ', flush=True)
	return c, net.single_descent()

def worker(q, l):
	init_worker()
	while True:
		args = q.get()
		if args is None:
			break
		i, (*args) = args
		l[i] = single_example(*args)
		q.task_done()


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

def main():
	images = [None]*batch_size*num_batches
	labels = [None]*batch_size*num_batches
	for i in range(batch_size*num_batches):
		images[i], labels[i] = read_file(f'image_data/data/{i:05d}')

	num_processes = 8
	manager = SyncManager()
	manager.start(init_worker)
	lst = manager.list([None]*batch_size)
	write_q = JoinableQueue(batch_size)
	pool = Pool(num_processes, worker, (write_q, lst))
	for e in range(epochs):
		print(f'Epoch {e}')
		for batch in range(num_batches):
			print(f' - Batch {batch} -', end=' ', flush=True)
			gradW = [np.zeros(w.shape) for w in n.weights]
			gradB = [np.zeros(b.shape) for b in n.biases]
			offset = batch*batch_size
			cost = 0
			for b in range(batch_size):
				write_q.put((b, n, offset+b, images[offset+b], labels[offset+b]))
			try:
				write_q.join()
				for c, (w, b) in lst:
					cost += c
					for k in range(len(n.nodes)-1):
						gradW[k] += w[k]
						gradB[k] += b[k]
			except KeyboardInterrupt:
				return
			print('-', cost/batch_size)
			n.descend_batch(gradW, gradB, batch_size)
		print('\n')
		n.save()

	for i in range(num_processes):
		write_q.put(None)

	pool.close()
	pool.join()

if __name__ == '__main__':
	main()