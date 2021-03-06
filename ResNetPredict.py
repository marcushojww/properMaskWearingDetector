import torch
from torchvision import transforms, datasets


import torch.nn as nn

loss = nn.CrossEntropyLoss()
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_height, img_width = 150, 150

transform = transforms.Compose([transforms.Resize(img_height),transforms.CenterCrop(img_width), transforms.ToTensor()])
input_shape=torch.Tensor((img_height, img_width))

model_resnet50 = torch.load(r'./RNoutput/RNmodel.pth')
testimg = Image.open(r'./googleTestImages/with_mask/test5.jpg')
# testimg = Image.open(r'./test/incorrect_mask/00003_Mask_Mouth_Chin.jpg')
testimg = transform(testimg)
testimg=testimg[None,:].to(device) #trick to add one more dimension (batch dimension)

model_resnet50.eval()
with torch.no_grad():
  pred_prob = torch.softmax(model_resnet50(testimg),dim=1)
  print(pred_prob)
  pred_class = pred_prob.argmax(1)

if pred_class == 0:
    prediction = 'improper mask'
    print("This image is "+ prediction + ".")
    print('With probability of ' + str(pred_prob[0,0].item()))
elif pred_class == 1:
    prediction = 'with mask'
    print("This image is "+ prediction + ".")
    print('With probability of ' + str(pred_prob[0,1].item()))
else:
    prediction = 'without mask'
    print("This image is "+ prediction + ".")
    print('With probability of ' + str(pred_prob[0,2].item()))