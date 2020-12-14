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

#read train
train = pd.read_csv("train.csv")
print(train.shape) # (42000, 785) - (image, column)

#read test 
test= pd.read_csv("test.csv")
print(test.shape) #(28000, 784) | 785-1 olması sebebi labelin olmaması
#label = resimde yazan sayının tutulduğu kısım. Resimdeki sayı 3 se label 3 tür

#Y_train and X_train
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
#train datamdan, label dışında kalan bütün kısmınları alıp X_train diyorum

#Test data set
img = X_train.iloc[41200].to_numpy() 
# 41200. resmi alıp matrixe çevirdim
img = img.reshape((28,28))
# 28 e 28 lik bir matrix haline geldi
plt.imshow(img,cmap='gray')
plt.title(train.iloc[41200,0])
plt.axis("off")
plt.show()

