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
#import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
#from matplotlib import rc
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class_names = []

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



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # in_channels = 1 because they are greyscale images
        # out_channels = 6 means, we're using 6, 5*5 filters/kernals, thus 6 outputs will be there
        # output of the previous layer is the input to the next layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # when moving from the convolutional layer to fully connected layers, inputs should be flattened
        self.fc1 = nn.Linear(in_features=12 * 53 * 53, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        # out_features = 15 because we have 15 class labels
        self.out = nn.Linear(in_features=60, out_features=15)

    def forward(self, t):
        # (1) input layer
        t = t  # here we show this for clarity

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 53 * 53)  # change the shape accordingly
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

def partition_dataset():
    # change to test dataset
    global class_names
    root_data = '/s/bach/b/class/cs535/cs535a/test-categorized/'
    dataset = torchvision.datasets.ImageFolder(root_data,
                                   transform=transforms.Compose([
                                       transforms.Resize(size=(224, 224)),
                                       transforms.Grayscale(1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    class_names = dataset.classes

    print('Test Dataset Transformed')
    size = dist.get_world_size()
    bsz = int(1024 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    print('Partition completed')
    test_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)
    print('Test Data Loaded')
    return test_set, bsz


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

@torch.no_grad()  # because we don't need this function to track gradients
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    for batch in loader:
        images, labels = batch
        #print("Labels inside get_all_preds:")
        #print(labels)
        #print("Dtype of labels :")
        #print(labels.type())
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds), dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels.float()),dim=0
        )

    return all_preds, all_labels

def gather(tensor, tensor_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
    #print("Tensor type:")
    #print(tensor.type())
    rank = dist.get_rank()
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert(tensor_list is not None)
        dist.gather(tensor, gather_list=tensor_list, group=group)
    else:
        dist.gather(tensor, dst=root, group=group)


def run(rank, size):
    torch.manual_seed(1234)
    test_set, bsz = partition_dataset()

    model = load_model(nn.parallel.DistributedDataParallel(Net()), "sgd_150_0.1_state_Dict_150.pth").float()

    num_batches = np.ceil(len(test_set.dataset) / float(bsz))
    best_loss = float("inf")

    preds, labels = get_all_preds(model, test_set)
    #print("Preds Size")
    #print(preds.size())  ([7551,15])
    #print("Labels Size")  ([7551])
    #print(labels.size())

    pred_lbl_fl = preds.argmax(1).float()
    lbl_fl = labels.float()

    prediction_list = [torch.zeros_like(pred_lbl_fl) for _ in range(size)]
    labels_list = [torch.zeros_like(pred_lbl_fl) for _ in range(size)]

    #print(labels)
    if dist.get_rank() == 0:
        gather(pred_lbl_fl, prediction_list)
        gather(lbl_fl, labels_list)
    else:
        gather(pred_lbl_fl)
        gather(lbl_fl)

    if dist.get_rank() == 0:

        new_preds = torch.tensor([], dtype=torch.float32)
        new_labels = torch.tensor([], dtype=torch.float32)
        for t1 in prediction_list:
            new_preds = torch.cat((new_preds,t1), dim=0)

        for t2 in labels_list:
            new_labels = torch.cat((new_labels,t2),dim=0)
            
        print("Preds:")
        k = new_preds.tolist()
        print(k[0:20])
        print("Actual:")
        j = new_labels.tolist()
        print(j[0:20])
		
        accuracry = calculate_accuracy(new_labels, new_preds)
        print("Accuracy : ",accuracry)
        print("Classification Report")
        print(classification_report(new_labels, new_preds, target_names=class_names))

        #roc_auc = roc_auc_compute_fn(new_preds, new_labels)
        #print("ROC-AUC score :", roc_auc)

        cm = get_confusion_matrix(new_labels, new_preds)
        print("Confusion Matrix :")
        print(cm)

def get_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    #df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    #hmap = sns.heatmap(df_cm, annot=True, fmt="d")
    #hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    #hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    return cm

def calculate_accuracy(y_true, y_pred):
  return (y_true == y_pred).sum().float() / len(y_true)

def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return roc_auc_score(y_true, y_pred)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'madison'
    os.environ['MASTER_PORT'] = '17578'

    # initialize the process group
    # When using DistributedDataParallel, it's very important to give a sufficient timeout because fast processes might arrive early and timeout on waiting for stragglers
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size), init_method='tcp://madison:23978',timeout=datetime.timedelta(weeks=120))

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)



if __name__ == "__main__":
    try:
        setup(sys.argv[1], sys.argv[2])
        print(socket.gethostname() + ": Setup completed!")

        run(int(sys.argv[1]), int(sys.argv[2]))

    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
