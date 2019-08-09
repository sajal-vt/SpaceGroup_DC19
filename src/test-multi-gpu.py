"""
Optional: Data Parallelism
==========================
**Authors**: `Sung Kim <https://github.com/hunkim>`_ and `Jenny Kang <https://github.com/jennykang>`_

In this tutorial, we will learn how to use multiple GPUs using ``DataParallel``.

It's very easy to use GPUs with PyTorch. You can put the model on a GPU:

.. code:: python

    device = torch.device("cuda:0")
    model.to(device)

Then, you can copy all your tensors to the GPU:

.. code:: python

    mytensor = my_tensor.to(device)

Please note that just calling ``my_tensor.to(device)`` returns a new copy of
``my_tensor`` on GPU instead of rewriting ``my_tensor``. You need to assign it to
a new tensor and use that tensor on the GPU.

It's natural to execute your forward, backward propagations on multiple GPUs.
However, Pytorch will only use one GPU by default. You can easily run your
operations on multiple GPUs by making your model run parallelly using
``DataParallel``:

.. code:: python

    model = nn.DataParallel(model)

That's the core behind this tutorial. We will explore it in more detail below.
"""


######################################################################
# Imports and parameters
# ----------------------
#
# Import PyTorch modules and define parameters.
#
from __future__ import print_function
from __future__ import division
from torch.utils import data
import h5py
from hdf5_dataset import HDF5Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn as nn

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

num_epochs = 10
loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 30}
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100


######################################################################
# Device
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Dummy DataSet
# -------------
#
# Make a dummy (random) dataset. You just need to implement the
# getitem
#

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


######################################################################
# Simple Model
# ------------
#
# For the demo, our model just gets an input, performs a linear operation, and
# gives an output. However, you can use ``DataParallel`` on any model (CNN, RNN,
# Capsule Net etc.)
#
# We've placed a print statement inside the model to monitor the size of input
# and output tensors.
# Please pay attention to what is printed at batch rank 0.
#

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


######################################################################
# Create Model and DataParallel
# -----------------------------
#
# This is the core part of the tutorial. First, we need to make a model instance
# and check if we have multiple GPUs. If we have multiple GPUs, we can wrap
# our model using ``nn.DataParallel``. Then we can put our model on GPUs by
# ``model.to(device)``
#

#model = Model(input_size, output_size)

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 231

# Batch size for training (change depending on how much memory you have)
batch_size = 600

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        #set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 512

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)





if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model_ft)

model.to(device)


######################################################################
# Run the Model
# -------------
#
# Now we can see the sizes of input and output tensors.
#
loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 30}

dataset = HDF5Dataset('/gpfs/alpine/world-shared/gen011/sajal/dc19/test', recursive=False, load_data=False, 
   data_cache_size=4, transform=None)

print(dataset)



data_loader = data.DataLoader(dataset)
print(data_loader)

def train_model(model, dataloaders, dataloader, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #data_loader = data.DataLoader(dataset)
        #dataloaders['train'] = data_loader
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                print("Input-size:", inputs.size())
                print("Output-size:", labels.size())
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

dataloaders_dict = {}
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model, hist = train_model(model, dataloaders_dict, data_loader, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


for data in data_loader:
    input = data.to(device)
    #output = model(input)
    print("Outside: input size", input.size())


