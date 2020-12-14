#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:36:22 2020

@author: alikemalcelenk
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
#import warnings
import warnings
#filter warnings
warnings.filterwarnings('ignore')

#TOTAL 70.000 RESMIM VAR. 42000 TRAIN, 28000 TEST

#Read Train
train = pd.read_csv("train.csv")
print(train.shape) # (42000, 785) - (image, column)

#Read Test 
test= pd.read_csv("test.csv")
print(test.shape) #(28000, 784) | 785-1 olması sebebi labelin olmaması
#label = resimde yazan sayının tutulduğu kısım. Resimdeki sayı 3 se label 3 tür

#Y_train and X_train
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
#train datamdan, label dışında kalan bütün kısmınları alıp X_train diyorum

#Test Data Set
img = X_train.iloc[41200].to_numpy() 
# 41200. resmi alıp matrixe çevirdim
img = img.reshape((28,28))
# 28 e 28 lik bir matrix haline geldi
plt.imshow(img,cmap='gray')
plt.title(train.iloc[41200,0])
plt.axis("off")
plt.show()

#Normalization
X_train = X_train / 255.0
test = test / 255.0
#normalizationın tam formülü bu değildir fakat resimlerle çalıştığımızda 255 yeterli
print("x_train shape: ",X_train.shape) #(42000, 784)
print("test shape: ",test.shape)  #(42000, 784) 

#Reshape
#resimleri 28x28x1 formatına getirip 3D yapmak çünkü keras bu formatta çalışıyor 
#x1 = gray scale
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape) #(42000, 28, 28, 1)
print("test shape: ",test.shape) #(28000, 28, 28, 1)

#Label Encoding
#0 => [1,0,0,0,0,0,0,0,0,0]
#9 => [0,0,1,0,0,0,0,0,0,1]
from keras.utils.np_utils import to_categorical  #one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)


# Split Train and Validation
# Traini 2 ye ayırdım. %90 train %10 validation. Val ile testleri yapıcam
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2) 
print("x_train shape",X_train.shape) #(37800, 28, 28, 1)
print("x_test shape",X_val.shape) #(4200, 28, 28, 1)
print("y_train shape",Y_train.shape) #(37800, 10)
print("y_test shape",Y_val.shape) #(4200, 10)




