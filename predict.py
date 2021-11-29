from PIL import Image
import matplotlib
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

# choose image file path
# im = Image.open(r'./predict/mask.jpg')

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = transforms.Resize(size=(config.INPUT_HEIGHT, config.INPUT_WIDTH))
predictTransform = transforms.Compose([resize, transforms.ToTensor()])
# predictImage = predictTransform(im)
# predictImage = predictImage[None,:].to(device)

# visualize data
# matplotlib.use('tkagg')
# oneChannelPredictImage = predictImage[0]
# plt.imshow(oneChannelPredictImage)
# plt.show()

predictDataset = ImageFolder(root=config.PREDICT, transform=predictTransform)
predictDataLoader = DataLoader(predictDataset, batch_size=1)

# load saved model
model = torch.load('./output/model.pth')
model.eval()

with torch.no_grad():
    for (image, label) in predictDataLoader:
        image = image.to(device)
        # obtain prediction
        pred = model(image)
        # compare predictions results for both classes

        if (pred[0][0] > pred[0][1]): #predicted with_mask
            print("[RESULT] Model predicted with mask")
        else:
            print("[RESULT] Model predicted without mask")
    