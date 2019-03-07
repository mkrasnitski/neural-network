from tkinter import *
from PIL import Image, ImageDraw
from network import Network
import numpy as np

class cv:
	def __init__(self):
		self.last_loc = None
		self.drag_loc = None
		self.width = 150
		self.height = 150
		self.network = Network('network')

		self.master = Tk()
		self.c = Canvas(self.master, width=self.width, height=self.height)
		self.c.pack()
		s = Button(self.master, text="Test", command=self.test)
		s.pack(side=LEFT)
		cl = Button(self.master, text="Clear", command=self.clear)
		cl.pack(side=RIGHT)
		self.c.bind("<Button-1>", self.click)
		self.c.bind("<B1-Motion>", self.draw)
		self.c.bind("<Button-3>", self.set_reference)
		self.c.bind("<B3-Motion>", self.move_window)
		self.clear()
		mainloop()

	def clear(self):
		self.c.delete('all')
		self.image = Image.new('L', (28, 28))
		self.draw = ImageDraw.Draw(self.image)

	def test(self):
		# self.image.save('image.png')
		data = np.array(list(self.image.getdata()))/255
		self.network.run(data, [0]*10)
		# print(self.network.get_guess())
		# self.network.print_last()
		print(self.network.get_guess())
		self.network.print_last()

	def click(self, event):
		self.last_loc = np.array([event.x, event.y])

	def set_reference(self, event):
		self.drag_loc = [event.x, event.y]

	def move_window(self, event):
		x_offset = event.x_root - 8 - self.drag_loc[0]
		y_offset = event.y_root - 31 - self.drag_loc[1]
		self.master.geometry(f'+{x_offset}+{y_offset}')

	def draw(self, event):
		p1 = [event.x-1, event.y-1]
		p2 = [event.x+1, event.y+1]
		# print(p1 + p2)
		self.c.create_oval(*p1, *p2, fill="black", width=5)

		new_loc = np.array([event.x, event.y])
		size = max(abs(np.subtract(self.last_loc, new_loc)))
		x_space = np.linspace(self.last_loc[0], new_loc[0], size)
		y_space = np.linspace(self.last_loc[1], new_loc[1], size)
		for x, y in zip(x_space, y_space):
			# print(x, y)
			p1 = (x-1,y-1)
			p2 = (x+1,y+1)
			self.c.create_oval(*p1, *p2, fill="black", width=5)
		self.draw.line([*(28*self.last_loc/150), *(28*new_loc/150)], fill="white", width=2)
		self.last_loc = new_loc.copy()

c = cv()