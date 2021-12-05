# properMaskWearingDetector

# GoogLeNet
## for testing of GoogLeNet model:
`python3 GoogLeNetTrain.py`

## for predicting GoogLeNet.model:
`python3 GoogLeNetPredict.py`
note: the image to be tested can be changed at line 15:
`testimg = Image.open(r'./googleTestImages/improper_mask/test2.jpg')`

The model is saved at GLNmodel/GLNmodel.pth
The plotted graph is saved at GLNmodel/GLNmodelPlot.png


# ResNet50
## for testing of ResNet50 model:
`python3 ResNetTrain.py`

## for predicting ResNet50 model:
`python3 ResNetPredict.py`
note: the image to be tested can be changed at line 15:
`testimg = Image.open(r'./googleTestImages/improper_mask/test2.jpg')`

The model is saved at RNmodel/RNmodel.pth
The plotted graph is saved at RNmodel/RNmodelPlot.png

# Predicting new dataset
trains a new dataset wwith current model with cmd:
`python3 prdictingDataset2.py` 
note: 
change your model by changing the code at line 27 `model = torch.load('./GLNoutput/GLNmodel.pth')`
change your dataset to train by changing the code at line35 `test_folder = 'test_2'`
