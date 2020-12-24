from random import randint
from time import sleep
import torch
import torch.distributed as dist
import os
import sys
import torchvision
import random
import numpy as np
import pandas as pd
import subprocess
import math
import socket
import traceback
import datetime
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from random import Random


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		# in_channels = 1 because they are greyscale images
		# out_channels = 6 means, we're using 6, 5*5 filters/kernals, thus 6 outputs will be there
		# output of the previous layer is the input to the next layer
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

		# when moving from the convolutional layer to fully connected layers, inputs should be flattened
		self.fc1 = nn.Linear(in_features=12 * 125 * 125, out_features=120)
		self.fc2 = nn.Linear(in_features=120, out_features=60)
		# out_features = 15 because we have 15 class labels
		self.out = nn.Linear(in_features=60, out_features=15)

	def forward(self, t):
		# (1) input layer
		t = t    # here we show this for clarity

		# (2) hidden conv layer
		t = self.conv1(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		# (3) hidden conv layer
		t = self.conv2(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		# (4) hidden linear layer
		t = t.reshape(-1,12 * 125 * 125)   # change the shape accordingly
		t = self.fc1(t)
		t = F.relu(t)

		# (5) hidden linear layer
		t = self.fc2(t)
		t = F.relu(t)

		# (6) output layer
		t = self.out(t)
		# softmax returns a probability of predictions for each class,
		# however, we don't need this, if we're using cross_entropy during training
		# t = F.softmax(t, dim=1)

		return t

class Partition(object):

	def __init__(self, data, index):
		self.data = data
		self.index = index

	def __len__(self):
		return len(self.index)

	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]

class DataPartitioner(object):

	def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
		self.data = data
		self.partitions = []
		rng = Random()
		rng.seed(seed)
		data_len = len(data)
		indexes = [x for x in range(0, data_len)]
		rng.shuffle(indexes)

		for frac in sizes:
			part_len = int(frac * data_len)
			self.partitions.append(indexes[0:part_len])
			indexes = indexes[part_len:]

	def use(self, partition):
		return Partition(self.data, self.partitions[partition])

""" Partitioning MNIST """
def partition_dataset():

	root_data='/s/bach/b/class/cs535/cs535a/train-categorized/'
	dataset = torchvision.datasets.ImageFolder(root_data,
							 transform=transforms.Compose([
							   transforms.Resize(size=(512,512)),
							   transforms.Grayscale(1),
							   transforms.ToTensor(),
							   transforms.Normalize((0.1307,), (0.3081,))
							]))
	print('Dataset Transformed')
	size = dist.get_world_size()
	bsz = int(1024/ float(size))
	partition_sizes = [1.0 / size for _ in range(size)]
	partition = DataPartitioner(dataset, partition_sizes)
	partition = partition.use(dist.get_rank())
	print('Partition completed')
	train_set = torch.utils.data.DataLoader(partition,
										   batch_size=bsz,
										   shuffle=True)
	print('Data Loaded')
	return train_set, bsz

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
	# Print New Line on Complete
	if iteration == total:
		print()

""" Gradient averaging. """
def average_gradients(model):
	size = float(dist.get_world_size())
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
		param.grad.data /= size


""" Distributed Synchronous SGD Example """
def run(rank, size, algo, epochs, learning_rate):
	torch.manual_seed(1234)
	train_set, bsz = partition_dataset()
	model = Net().to(rank)

	ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

	if algo == 'sgd':
		optimizer = optim.SGD(ddp_model.parameters(),lr=learning_rate, momentum=0.5)
	elif algo == 'adam':
		optimizer = optim.Adam(ddp_model.parameters(),lr=learning_rate)

	#criterion = nn.cross_entropy()
	num_batches = np.ceil(len(train_set.dataset) / float(bsz))
	best_loss = float("inf")
	epoch_loss_list = []
	for epoch in range(epochs):
		epoch_loss = 0.0
		printProgressBar(0, len(train_set), prefix = 'Progress:', suffix = 'Complete', length = 50)
		for i, (data, target) in enumerate(train_set):
			#if torch.cuda.is_available():
			#	data, target = data.cuda(), target.cuda()

			#We need to clear them out before each instance
			optimizer.zero_grad()

			output = ddp_model(data)

			loss = F.cross_entropy(output, target)
			epoch_loss += loss.item()
			loss.backward()
			#average_gradients(model)
			optimizer.step()
			printProgressBar(i + 1, len(train_set), prefix = 'Progress:', suffix = 'Complete', length = 50)
		epoch_loss_list.append(float(epoch_loss/num_batches))
		# print in intervals
		if epoch % (epochs/10) == 0:
			print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
		# if ((dist.get_rank() == 0) and (epoch == 45 or epoch == 50 or epoch == 55 or epoch == 60 or epoch == 65 or epoch == 70)):
		# 	best_loss1 = epoch_loss / num_batches
		# 	path_name1 = algo + "_" + str(epoch) + "_" + str(learning_rate) + "_" + "model.pth"
		# 	torch.save(model.state_dict(), path_name1)
		# 	# torch.save(model.state_dict(), "best_model.pth")
		#
		# 	f1 = open("512_results_intermediate.txt", "a")
		# 	new_list1 = ['{:.2f}'.format(x) for x in epoch_loss_list]
		# 	list_epoch1 = ",".join(new_list1)
		# 	info1 = algo.upper() + "\t" + str(epochs) + "\t" + str(num_batches) + "\t" + str(learning_rate) + "\t" + str(best_loss1) + "\t" + list_epoch1 +"\n"
		# 	f1.write(info1)
		# 	f1.close()

	if dist.get_rank() == 0:
		best_loss = epoch_loss / num_batches
		path_name = algo + "_" + str(epochs) + "_" + str(learning_rate) + "_" + "ddp_model.pth"
		torch.save(ddp_model.state_dict(), path_name)

		f = open("512_results.txt", "a")
		new_list = ['{:.2f}'.format(x) for x in epoch_loss_list]
		list_epoch = ",".join(new_list)
		info = algo.upper() + "\t" + str(epochs) + "\t" + str(num_batches) + "\t" + str(learning_rate) + "\t" + str(best_loss) + "\t" + list_epoch +"\n"
		f.write(info)
		f.close()

def setup(rank, world_size):
	os.environ['MASTER_ADDR'] = 'albany'
	os.environ['MASTER_PORT'] = '20435'

	# initialize the process group
	# When using DistributedDataParallel, it's very important to give a sufficient timeout because fast processes might arrive early and timeout on waiting for stragglers
	dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size), init_method='tcp://albany:23402', timeout=datetime.timedelta(weeks=120))

	# Explicitly setting seed to make sure that models created in two processes
	# start from same random weights and biases.
	torch.manual_seed(42)

if __name__ == "__main__":
	try:
		setup(sys.argv[1], sys.argv[2])
		print(socket.gethostname()+": Setup completed!")

		algo_list = ['sgd']
		learning_rates = [ 1e-1]
		epoch_list = [1]

		for algo in algo_list:
			print(algo.upper())
			for epochs in epoch_list:
				for lr in learning_rates:
					run(int(sys.argv[1]), int(sys.argv[2]), algo, epochs, lr)

	except Exception as e:
		traceback.print_exc()
		sys.exit(3)