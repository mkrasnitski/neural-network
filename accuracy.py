from network import Network
import pickle
import numpy as np
import sys
from multiprocessing.dummy import Pool as ThreadPool

def read_file(path):
	image, label = None, None
	with open(path, 'rb') as f:
		image = pickle.load(f)
		label = pickle.load(f)
		labels = [1 if i == label else 0 for i in range(10)]
	return image, label, labels
path = 'network'
if len(sys.argv) > 1:
	path = sys.argv[1]
n = Network(path)

num = 5000

images = [None]*num
labels = [None]*num
y = [None]*num
for i in range(num):
	images[i], labels[i], y[i] = read_file(f'image_data/data/{i:05d}')

count = 0
for i in range(num):
	n.run(images[i], y[i])
	predicted = np.argmax(n.activations[-1])
	if predicted == labels[i]:
		count += 1
print(f'{count}/{num} = {100*count/num}% correct')