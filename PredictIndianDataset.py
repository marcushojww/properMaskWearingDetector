# set the matplotlib backend so figures can be saved in the background
import matplotlib
# matplotlib.use("Agg")

# import the necessary packages
from lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import config

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load saved model
model = torch.load('./GLNoutput/GLNmodel.pth')

# initialize our data augmentation functions
resize = transforms.Resize(size=(config.INPUT_HEIGHT,
 config.INPUT_WIDTH))

testTransforms = transforms.Compose([resize, transforms.ToTensor()])

test_folder = input("Enter 'test' or 'test_2' for model testing: ")

# initialize the training and validation dataset
print("[INFO] loading test dataset...")
testDataset = ImageFolder(root=test_folder, 
 transform=testTransforms)

print("[INFO] test dataset contains {} samples...".format(len(testDataset)))

testDataLoader = DataLoader(testDataset, batch_size=config.BATCH_SIZE)

# turn off autograd for testing evaluation
with torch.no_grad():
 # set the model in evaluation mode
 model.eval()
 
 # initialize a list to store our predictions
 preds = []

 # loop over the test set
 for (input, label) in testDataLoader:
  # send the input to the device
  input = input.to(device)

  # make the predictions and add them to the list
  pred = model(input)
  preds.extend(pred.argmax(axis=1).cpu().numpy())
  
# generate a classification report
print(classification_report(np.array(testDataset.targets),
 np.array(preds), target_names=testDataset.classes))