import sys
import math
# import matplotlib.pyplot as plt
from network import Network

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
	# if i % 1000 == 0:
	print(f'{i} - {x:.15f}')
	n.descend()
print()
# plt.plot(range(len(costs)), costs)
# plt.ylim(bottom=0)
# plt.savefig('figure.png')

n.print_last()