#Final Project Implementation
#Zach Ullom, Duncan Stephenson
#Due Dec 13th 2024

#Inmport Libraries

#Import libraries

#Tensorflow Libraries
import keras
from keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import models
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


#Pytorch libraries
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import timm
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

#Other libraries
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Library used for voting
from scipy.stats import mode


#LOADING DATASETS

#Arrays for image names and preprocessed images
AkLeaves = []
Ala_IdrisLeaves = []
BuzguluLeaves = []
DimnitLeaves = []
NazliLeaves = []

preprocessedLeaves = []
Labels = []

#initial path for reading in images
path1 = "Ak"
path2 = "Ala_Idris"
path3 = "Buzgulu"
path4 = "Dimnit"
path5 = "Nazli"

#Final Dataframe to show all scores
finalScores = pd.DataFrame()

#fill in missing values with NaN (not a number)
finalScores['Model Name'] = np.nan
finalScores['Accuracy'] = np.nan
finalScores['Precision'] = np.nan
finalScores['F1 Score'] = np.nan
finalScores['Recall'] = np.nan

#add model names to finalScores data frame
finalScores.loc[0, 'Model Name'] = 'VGG19'
finalScores.loc[1, 'Model Name'] = 'ResNetV2'
finalScores.loc[2, 'Model Name'] = 'ViT'
finalScores.loc[3, 'Model Name'] = 'DesNet201'
finalScores.loc[4, 'Model Name'] = 'ResneXT'
finalScores.loc[5, 'Model Name'] = 'Hard Voting'
finalScores.loc[6, 'Model Name'] = 'Soft Voting'



#Get all file names and save them to their respective arrays
#The result is an array of strings representing paths to each image
def getImageNames(arr1, startPath):

    #add on path as I search through the folder
    x = ''
    for img in os.listdir(startPath):


        #get image path for each image
        x = os.path.join(startPath, img)


        #add new data to the end of read training images array
        arr1.append(x)


#mean values as specified by the journal
means = (0.485, 0.456, 0.406)
stdev = (0.229, 0.224, 0.225)


#Read in images from the image path arrays, preprocess them as is specified in the journal, add them to an array of preprocessed images, and add new data to the labels array
def preprocessImage(arr1, arr2, leafName):


    #iterate through the entire array
    for i in range(len(arr1)):
                    
        #read in iamge
        img = cv2.imread(arr1[i])

        #resize the image to 256 x 256 as is specified by the journal
        img = cv2.resize(img, (256, 256))

        #center crop to 224
        #(256-224)/2 = 16, so starting coordinate is 16 end ending is 240
        img = img[16:240, 16:240]

        #Normalize image with stds and means specific in paper
        img = (img/ 255.0 - means) / stdev

        #add new preprocessed image to preprocessed images array
        arr2.append(img)

        #add new labels name to labels array
        Labels.append(leafName)

#function calls
getImageNames(AkLeaves, path1)
getImageNames(Ala_IdrisLeaves, path2)
getImageNames(BuzguluLeaves, path3)
getImageNames(DimnitLeaves, path4)
getImageNames(NazliLeaves, path5)

preprocessImage(AkLeaves, preprocessedLeaves, 'Ak')
preprocessImage(Ala_IdrisLeaves, preprocessedLeaves, 'Ala_Idris')
preprocessImage(BuzguluLeaves, preprocessedLeaves, 'Buzgulu')
preprocessImage(DimnitLeaves, preprocessedLeaves, 'Dimnit')
preprocessImage(NazliLeaves, preprocessedLeaves, 'Nazli')


#add gaussian noise and random erasing to preprocessed images
def gNoise(arr1):

    #iterate through the entire array
    for i in range(len(arr1)):

        #get current iteration from the array
        img1 = arr1[i]

        #change array to tensor
        transform1 = transforms.Compose([transforms.ToTensor()])
        
        #implement random erasing using the scale and ratio values specified by the journal
        transform2 = transforms.Compose([transforms.RandomErasing(p=0.5, scale = (.02, .32), ratio =(.3, 3.2 ))])

        #apply transformations
        img1 = transform1(img1)
        img1 = transform2(img1)

        #generate gaussian noise
        #mean = 0, std = 1
        noise = np.random.normal(0, 1, img1.shape)

        #combine noise and image
        img1 = np.clip(img1 + noise, 0, 1)

        #convert back to np array
        img1 = np.array(img1)

        #replace original preprocessed image with new gaussian noise image
        arr1[i] = img1

#function call for gaussian noise
gNoise(preprocessedLeaves)


#using one hot encoding to convert the labels array to categories
e1 = OneHotEncoder()

#use original labels to create new categories
Labels = pd.DataFrame(Labels)

#convert to array
x = e1.fit_transform(Labels).toarray()

#array used for new categorized labels
Labels2 = []

#add all one hot encoded data to labels 2 array
for j in range(500):
    Labels2.append(x[j])

#convert labels2 and the preprocessed images to a np array
Labels2 = np.array(Labels2)
preprocessedLeaves = np.array(preprocessedLeaves)


#Train Test Split
xTrain, xTest1, yTrain, yTest1 = train_test_split(preprocessedLeaves, Labels2, test_size= .20, train_size= .80, random_state= 42)

#10% validation
xVal, xTest, yVal, yTest = train_test_split(xTest1, yTest1, test_size= .5, train_size= .5, random_state= 42)



#FEATURE EXTRACTION

#preprocessedLeaves = np.array(preprocessedLeaves)


#VGG19
def runVGG19():

    #Import vgg19 pretrained model   
    from tensorflow.keras.applications import VGG19

    #create model for vgg19 with imagenet pretrained weights but without the top layer, so we can use it for our own classification task
    model1 = VGG19(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))

    #Freeze layers to modify output
    for layer in model1.layers:
        layer.trainable = False

    #custom layers to ouput 5 possible classifications. required one fully connected layer to change to an output of 5
    x = Flatten()(model1.output)
    x = Dense(256, activation = 'relu')(x)
    output = Dense(5, activation = 'softmax')(x)

    #since we're not using sequential, we have to manually plug in the input and output
    #the input is the pretrained model input, and the output is our change to 5 classes
    model2 = Model(inputs = model1.input, outputs = output) 

    #return the newly created model
    return model2



#RESNETV2
def runResNetV2():

    #Import ResnetV2
    from tensorflow.keras.applications import ResNet50V2

    #create a ResnetV2 pretrained model with the same parameters as vgg19
    model1 = ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))

    #Freeze Layers
    for layer in model1.layers:
        layer.trainable = False

    #custom layers to ouput 5 possible classifications
    x = Flatten()(model1.output)
    x = Dense(256, activation = 'relu')(x)
    output = Dense(5, activation = 'softmax')(x)

    #create model and return
    model2 = Model(inputs = model1.input, outputs = output) 
    return model2

#DESNET201 MODEL
def runDesnet():

    #import DesNet201 pretrained model
    from tensorflow.keras.applications import DenseNet201

    #create Densenet201 model with the same parameters as the other two
    model1 = DenseNet201(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))

    #Freeze Layers
    for layer in model1.layers:
        layer.trainable = False

    #custom layers to ouput 5 possible classifications
    x = Flatten()(model1.output)
    x = Dense(256, activation = 'relu')(x)
    output = Dense(5, activation = 'softmax')(x)

    #create and return model
    model2 = Model(inputs = model1.input, outputs = output) 
    return model2

#This function actually runs the models we've created. So the function calls will read runSpecificModel, then runModel
def runModel(model2, xTrain, yTrain, xTest, yTest, modelNum):

    #compile the model with a learning rate of 0.001 as specified by the journal, loss is cross entropy, and metrics are accuracy, precision, recall, and f1-score as is in the journal
    model2.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'f1_score', 'recall'])
    
    #change the shape of the iamge data to work with the model. data value is maintained
    xTrain = np.transpose(xTrain, (0, 2, 3, 1))
    xTest = np.transpose(xTest, (0, 2, 3, 1))

    #run the model with 9 epochs, and a batch size of 64 as is specified in the journal
    model2.fit(xTrain, yTrain, epochs = 9, batch_size = 64)

    #predict the testing data
    pred = model2.predict(xTest)

    #function call for getScore
    getScore(pred, yTest, modelNum)



    #return the predictions to use for voting methods
    return pred

#use the predictions and labels from each model to generate all metrics and "save them" to the final scores dataframe
def getScore(pred, yTest, modelNum):


    #convert raw predictions into a single number
    pred = np.argmax(pred, axis = 1)

    #turn each label into a single number
    yTest = np.argmax(yTest, axis = 1)

    #get metrics as specified by the journal
    f1Score = f1_score(yTest, pred, average = 'weighted')
    precisionScore = precision_score(yTest, pred, average = 'weighted')
    recallScore = recall_score(yTest, pred, average = 'weighted')
    accuracyScore = accuracy_score(yTest, pred)

    #add the metrics to the final scores dataframe coresponding to each of the models
    finalScores.loc[modelNum, 'Accuracy'] = accuracyScore
    finalScores.loc[modelNum, 'Precision'] = precisionScore
    finalScores.loc[modelNum, 'Recall'] = recallScore
    finalScores.loc[modelNum, 'F1 Score'] = f1Score




#RESNEXT MODEL
def runResNext(xTrain, yTrain, xTest, yTest):

    #create a pretrained resnext model
    model = timm.create_model('resnext50_32x4d', pretrained = True)
    
    #Freeze layers in the model
    for param in model.parameters():
        param.requires_grad = False

    #make only 5 possible classifications
    model.fc = nn.Linear(model.fc.in_features, 256)
    model.fc = nn.Linear(model.fc.in_features, 5)

    #Set model to evaluation mode
    model.eval()

    #define loss function as cross entropy
    calcLoss = nn.CrossEntropyLoss()

    #specify optimizer as Adam with a learning rate of 0.001, same asthe tensorflow models
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    #convert all arrays to tensors
    xTrain = torch.tensor(xTrain)
    yTrain = torch.tensor(yTrain)

    xTest = torch.tensor(xTest)
    yTest = torch.tensor(yTest)


    #create a dataloader with a batch size of 64. Originally there was a data type error, thats why xTrain is converted to a float
    dataset = TensorDataset(xTrain.float(), yTrain)
    dataloader = DataLoader(dataset, batch_size = 64)

    #change the model to training mode
    model.train()

    #Train Dataset with 9 epochs
    for epoch in range(9):

        #iterate through the dataloader
        for i, (inputs, labels) in enumerate(dataloader):

            #reset gradients
            optimizer.zero_grad()

            #plug in the current iteration input to the model
            outputs = model(inputs)

            #calculate loss based on the output of the model and the labels
            loss = calcLoss(outputs, labels)

            #print the current loss so we can see it
            print(loss.item())

            #compute the gradient of the loss
            loss.backward()

            #update the parameters using the Adam otpimizer specified above
            optimizer.step()


    #set back to evaluation mode
    model.eval()

    #disable gradient calculation and predict the testing data
    with torch.no_grad():
        pred = model(xTest.float())

    #softmax the predictions so we can have positive data for easy input into the getScore function
    predictions = torch.nn.functional.softmax(pred, dim=1)

    #function call for getScore
    getScore(predictions, yTest, 4)

    #return the predictions to use for voting methods
    return predictions

#ViT MODEL
def runViT(xTrain, yTrain, xTest, yTest):


    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    #create a pretrained ViT model
    model = timm.create_model('vit_base_patch16_224', pretrained = True)

    #Freeze layers in the model
    for param in model.parameters():
        param.requires_grad = False

    #change from 1000 outputs to 5
    model.head = nn.Linear(model.head.in_features, 256)
    model.head = nn.Linear(model.head.in_features, 5)


    #Set model to evaluation mode
    model.eval()

    #define loss function as cross entropy
    calcLoss = nn.CrossEntropyLoss()

    #define optimizer with learning rate of 0.001, as is specified by the journal
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    #convert dataset to tensors
    xTrain = torch.tensor(xTrain)
    yTrain = torch.tensor(yTrain)

    xTest = torch.tensor(xTest)
    yTest = torch.tensor(yTest)

    #create dataloader with batch size of 64
    dataset = TensorDataset(xTrain.float(), yTrain)
    dataloader = DataLoader(dataset, batch_size = 64)

    #set model to training mode
    model.train()

    #Train Dataset
    for epoch in range(9):

        #iterate through the dataset
        for i, (inputs, labels) in enumerate(dataloader):


            #reset gradients
            optimizer.zero_grad()

            #plug in current iteration of xTrain to the model
            outputs = model(inputs)

            #calculate loss of current iteration using the outputs and current label
            loss = calcLoss(outputs, labels)

            #print current loss function so we can see
            print(loss.item())

            #compute gradient of the loss
            loss.backward()

            #update parameters in the optimizer
            optimizer.step()


    #set model back to evaluation mode
    model.eval()


    #Testing the model

    #disable gradient calculation and predict the testing dataset
    with torch.no_grad():
        pred = model(xTest.float())

    #use softmax for positive values
    predictions = torch.nn.functional.softmax(pred, dim=1)

    #function call for getScore
    getScore(predictions, yTest, 2)

    #return predictions for voting methods
    return predictions

#HARD AND SOFT VOTING
def Voting(yTest):


    #function calls to run all models. they will each return predictions
    pred1 = runResNext(xTrain, yTrain, xTest, yTest)

    pred2 = runViT(xTrain, yTrain, xTest, yTest)

    pred3 = runVGG19()
    pred3 = runModel(pred3, xTrain, yTrain, xTest, yTest, 0)

    pred4 = runResNetV2()
    pred4 = runModel(pred4, xTrain, yTrain, xTest, yTest, 1)

    pred5 = runDesnet()
    pred5 = runModel(pred5, xTrain, yTrain, xTest, yTest, 3)


    #convert all predictions into np arrays
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    pred3 = np.array(pred3)
    pred4 = np.array(pred4)
    pred5 = np.array(pred5)

    #stack predictions to make a collective set of predictions
    predictions = np.stack([pred1, pred2, pred3, pred4, pred5], axis = 0)

    #HARD VOTING, a majority rule using the mode
    totalPredictions, _ = mode(predictions, axis = 0)

    #convert to 1D array
    totalPredictions = totalPredictions.squeeze()

    #convert hard voting predictions and labels to tensors
    totalPredictions = torch.tensor(totalPredictions)
    yTest = torch.tensor(yTest)

    #getScore function call
    getScore(totalPredictions, yTest, 5)

    #convert stacked predictions and labels to tensors
    predictions = torch.tensor(predictions)
    yTest = torch.tensor(yTest)

    #SOFT VOTING, using the average (or mean) of all predictions instead of the mode
    averagePredictions = torch.mean(predictions, dim = 0)

    #getScore function call
    getScore(averagePredictions, yTest, 6)

    #print the final scores of all models and voting methods
    print(finalScores)

#function call for voting and entire program    
Voting(yTest)

