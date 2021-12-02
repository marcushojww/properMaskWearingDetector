import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchsummary import summary
import torch
import time
import numpy as np
from sklearn.metrics import classification_report

img_height, img_width = 150, 150
epoch = 0
input_shape=torch.Tensor((img_height, img_width))

train_data_dir = 'train'
validation_data_dir = 'val'
test_data_dir = 'test'

classes = ('With mask','improper mask','Without mask')
num_classes = len(classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_resnet50=models.resnet50(pretrained=True).to(device)

for param in model_resnet50.layer1.parameters(): #disabling gradient in the 'features' part of the model (convolutional part)
    param.require_grad = False #tell the model to keep the weights fixed

for param in model_resnet50.layer2.parameters(): #disabling gradient in the 'features' part of the model (convolutional part)
    param.require_grad = False #tell the model to keep the weights fixed

for param in model_resnet50.layer3.parameters(): #disabling gradient in the 'features' part of the model (convolutional part)
    param.require_grad = False #tell the model to keep the weights fixed

for param in model_resnet50.layer4.parameters(): #disabling gradient in the 'features' part of the model (convolutional part)
    param.require_grad = False #tell the model to keep the weights fixed

num_features = model_resnet50.fc.in_features #get the number of inputs for the very last layer
model_resnet50.fc = nn.Linear(num_features, num_classes).to(device) # Replace the final classification layer
# features = list(model_resnet50.fc.children())[:-1] # Replace the final classification layer
# features.extend([nn.Linear(num_features, 2)])
# features.extend([nn.Softmax(dim=1)])

# model_resnet50.fc = nn.Sequential(*features).to(device)

# print(model_resnet50)

# Resize the image and do a center crop, and store them on a tensor
transform = transforms.Compose([transforms.Resize(img_height),transforms.CenterCrop(img_width), transforms.ToTensor()])

# Load the training and validation dataset
train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
val_dataset = datasets.ImageFolder(validation_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)

batch_size = 30
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=2)

print("[INFO] training dataset contains {} samples...".format(
	len(train_dataset)))
print("[INFO] validation dataset contains {} samples...".format(
	len(val_dataset)))
print("[INFO] test dataset contains {} samples...".format(
	len(test_dataset)))

augtransforms=transforms.Compose([transforms.Resize((150,150)),
transforms.RandomAffine(degrees=(-25,25),translate=(0.1,0.1),shear=(-7,7)),
transforms.RandomResizedCrop((150,150),scale=(0.8,1),ratio=(0.9,1.1)),
transforms.RandomHorizontalFlip(),
transforms.ToTensor()]) # don't forget to add ToTensor when using the DataLoader

batch_size = 30
augtrain_dataset = datasets.ImageFolder(train_data_dir, transform=augtransforms) # our new augmented dataset

batch_size=100

train_dataloader_forresnet50 = DataLoader(augtrain_dataset, batch_size=batch_size,shuffle=True,num_workers=2)
val_dataloader_forresnet50 = DataLoader(val_dataset, batch_size=batch_size,shuffle=False,num_workers=2)
test_dataloader_forresnet50 = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,num_workers=2)

# print(model_resnet50)
# def extractFeatures(model_resnet50,dataloader):
#     model_resnet50.eval()
#     output=torch.zeros((len(dataloader.dataset),2048,1,1))
#     label=torch.zeros((len(dataloader.dataset)))
#     i=0

#     with torch.no_grad():
#         for j, data in enumerate(dataloader):
#             inputs, labels = data
#             inputs=inputs
#             outputs=model_resnet50.maxpool(model_resnet50.relu(model_resnet50.bn1(model_resnet50.conv1(inputs))))
#             outputs=model_resnet50.layer1(outputs)
#             outputs=model_resnet50.layer2(outputs)
#             outputs=model_resnet50.layer3(outputs)
#             outputs=model_resnet50.layer4(outputs)
#             outputs=model_resnet50.avgpool(outputs)
#             output[i:i+outputs.shape[0],:,:,:]=outputs
#             label[i:i+labels.shape[0]]=labels
#             i=i+outputs.shape[0]
#             print(i)

#     return output, label

# train_feats, train_labels=extractFeatures(model_resnet50,train_dataloader_forresnet50)

# torch.save((train_feats,train_labels),'./ResNet_Features/train_data.pt')

# val_feats, val_labels=extractFeatures(model_resnet50,val_dataloader_forresnet50)

# torch.save((val_feats,val_labels),'./ResNet_Features/val_data.pt')

train_feats,train_labels=torch.load('./ResNet_Features/train_data.pt')
val_feats,val_labels=torch.load('./ResNet_Features/val_data.pt')

optimizer = torch.optim.RMSprop(model_resnet50.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()

batch_size=30
train_feat_dataset = torch.utils.data.TensorDataset(train_feats,train_labels)
train_feat_dataloader = DataLoader(train_feat_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

val_feat_dataset = torch.utils.data.TensorDataset(val_feats,val_labels)
val_feat_dataloader = DataLoader(val_feat_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

def train_resnet50_classifier(modelclassifier,dataloader,val_dataloader):
    modelclassifier.train() #set the mode of the model to train mode
    total_acc, total_count, rep_acc, rep_count = 0, 0, 0, 0
    train_loss, rep_loss=0.0, 0.0
    log_interval = 30
    start_time = time.time()
    # print("training...")
    for i, data in enumerate(dataloader):
        inputs, labels = data
        labels=labels.long()
        # labels=labels.reshape(labels.shape[0],1).float()

        optimizer.zero_grad()
        # forward propagation
        predicted_label = torch.softmax(modelclassifier(torch.flatten(inputs,start_dim=1)),dim=1)
        # calculate loss and backpropagate to model paramters
        loss = criterion(predicted_label, labels)
        train_loss += loss.item() # to be reported per epoch
        rep_loss += loss.item() # to be reported per log_interval batches
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelclassifier.parameters(), 0.1)
        # update parameters by stepping the optimizer
        optimizer.step()
        total_acc += ((predicted_label.argmax(1)) == labels).sum().item() # to be reported per epoch
        rep_acc += ((predicted_label.argmax(1)) == labels).sum().item() # to be reported per log_interval batches
        total_count += labels.size(0)
        rep_count += labels.size(0)
        if i % log_interval == 0 and i > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| train_loss {:8.3f} | accuracy {:8.3f} | elapsed time {:5.1f} seconds'.format(epoch, i, len(dataloader),
                                              rep_loss/log_interval, rep_acc/rep_count, elapsed))
            rep_acc, rep_count = 0, 0
            rep_loss=0.0
            start_time = time.time()


          # Make a pass over the validation data.
    # print("validating...")
    val_acc, val_count = 0.0, 0.0
    cum_loss = 0.0
    start_time = time.time()
    modelclassifier.eval()

    with torch.no_grad():
        for j, data in enumerate(val_dataloader):
            inputs, labels = data
            labels=labels.long()
            # labels=labels.reshape(labels.shape[0],1).float()

            # Forward pass. (Prediction stage)
            scores = torch.softmax(modelclassifier(torch.flatten(inputs,start_dim=1)),dim=1)
            cum_loss += criterion(scores, labels).item()

            # Count how many correct in this batch.
            val_acc += ((scores.argmax(1)) == labels).sum().item()
            val_count += labels.size(0)

    elapsed = time.time() - start_time
    # Logging the current results on validation.
    print('Validation-epoch %d. Avg-Val Loss: %.4f, Val Accuracy: %.4f,   Elapsed Time: %.1f' % 
            (epoch, cum_loss / (j + 1), val_acc/val_count,elapsed))
    return total_acc/total_count, val_acc/val_count, train_loss/(i+1), cum_loss / (j + 1)


# summary(model_resnet50,(3,img_height,img_width)) # using torchsummary

def evaluate_resnet50(model,dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    class0_count,class0_correct = 0,0
    class1_count,class1_correct = 0,0
    class2_count,class2_correct = 0,0
    wrong=torch.zeros((len(dataloader.dataset)),1)
    j=0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            labels=labels.long()
            # labels=labels.reshape(labels.shape[0],1).float()
            predicted_label = torch.softmax(model(inputs),dim=1)
            correct=(predicted_label.argmax(1) == labels).type(
			torch.float)
            correct=correct.reshape(correct.shape[0],1).int()
            # print(correct)

            for k, x in enumerate(labels):
                if(x.data==0):
                    class0_count += 1.0
                    if(correct[k].item() == True):
                        class0_correct += 1.0
                elif(x.data==1):
                    class1_count += 1.0
                    if(correct[k].data[0] == True):
                        class1_correct += 1.0
                else:
                    class2_count += 1.0
                    if(correct[k].data[0] == True):
                        class2_correct += 1.0
            total_acc += correct.sum().item()
            total_count += labels.size(0)
            wrong[j:j+labels.size(0),:]=~correct
            j=j+labels.size(0)
    class0_accuracy = class0_correct/class0_count
    class1_accuracy = class1_correct/class1_count
    class2_accuracy = class2_correct/class2_count
    return total_acc/total_count, wrong, class0_accuracy,class1_accuracy, class2_accuracy

ta=[]
va=[]
tl=[]
vl=[]

for epoch in range(40):
  train_acc, val_acc, train_loss, val_loss=train_resnet50_classifier(model_resnet50.fc,train_feat_dataloader,val_feat_dataloader)
  ta.append(train_acc)
  va.append(val_acc)
  tl.append(train_loss)
  vl.append(val_loss)

accuracy,wrong,class0_accuracy,class1_accuracy,class2_accuracy = evaluate_resnet50(model_resnet50,test_dataloader_forresnet50)
wrong_index = torch.where(wrong==1) # get indeces of the wrong predictions
wrong_index = wrong_index[0]
print(wrong_index)
print("Overall test accuracy: %.4f " % accuracy)
print("With mask test accuracy: %.4f " %  class0_accuracy)
print("Improper mask test accuracy: %.4f " % class1_accuracy)
print("Without mask test accuracy: %.4f " % class2_accuracy)


with torch.no_grad():
	# set the model in evaluation mode
	model_resnet50.eval()
	
	# initialize a list to store our predictions
	preds = []

	# loop over the test set
	for (x, y) in test_dataloader_forresnet50:
		# send the input to the device
		x = x.to(device)

		# make the predictions and add them to the list
		pred = torch.softmax(model_resnet50(x),dim=1)
		preds.extend(pred.argmax(axis=1).cpu().numpy())



# generate a classification report
print(classification_report(np.array(test_dataset.targets),
	np.array(preds), target_names=test_dataset.classes))


def plotResults(): 
    plt.figure(figsize=(10,6))
    epovec=range(len(ta))
    plt.plot(epovec,ta,epovec,tl,epovec,va,epovec,vl,linewidth=3)
    plt.legend(('Train_acc','Train_loss','Val_acc','Val_loss'))

    # make the graph understandable: 
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.show()
    plt.savefig("RNoutput/RNModelPlot")
plotResults()

torch.save(model_resnet50, "RNoutput/RNmodel.pth")