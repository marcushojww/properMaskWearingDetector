# set the device we will be using to train the model
from PIL import Image
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# img = Image.open(r'./train/incorrect_mask/06002_Mask_Mouth_Chin.jpg')

# initialize our data augmentation functions
# resize = transforms.Resize(size=(config.INPUT_HEIGHT,
# 	config.INPUT_WIDTH))
# affine = transforms.RandomAffine(degrees=(-30,30), translate=(0.1,0.1))
# hFlip = transforms.RandomHorizontalFlip(p=0.4)

# initialize our data augmentation functions
resize = transforms.Resize(size=(config.INPUT_HEIGHT,
	config.INPUT_WIDTH))
hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)

# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([resize,hFlip,vFlip,rotate, transforms.ToTensor()])

for i in range(5):
    newImg = trainTransforms(img)
    newImg = newImg.permute(1,2,0)
    imgplot = plt.imshow(newImg)
    plt.show()