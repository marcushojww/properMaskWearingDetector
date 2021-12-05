# properMaskWearingDetector

# GoogLeNet
## for testing of GoogLeNet model:
python3 GoogLeNetTrain.py

## for predicting GoogLeNet.model:
python3 GoogLeNetPredict.py
note: the image to be tested can be changed at line 15:
testimg = Image.open(r'./googleTestImages/improper_mask/test2.jpg')

The model is saved at GLNmodel/GLNmodel.pth
The plotted graph is saved at GLNmodel/GLNmodelPlot.png


# ResNet50
## for testing of ResNet50 model:
python3 ResNetTrain.py

## for predicting ResNet50 model:
python3 ResNetPredict.py
note: the image to be tested can be changed at line 15:
testimg = Image.open(r'./googleTestImages/improper_mask/test2.jpg')

The model is saved at RNmodel/RNmodel.pth
The plotted graph is saved at RNmodel/RNmodelPlot.png
