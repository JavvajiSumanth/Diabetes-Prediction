#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m

#import the dataset
dataset=pd.read_csv('diabetes.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

#input layer
classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=8))

#input layer
classifier.add(Dense(output_dim=4,init='uniform',activation='relu'))

#input layer
classifier.add(Dense(output_dim=1,init='uniform',activation='relu'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,Y_train,nb_epoch=100,batch_size=8)


#predicting the test set results
Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

s=int(m.sqrt(cm.size))
sum1=0
sum2=0 

for i in range(0,s):
    for j in range(0,s):
            if i==j:
                sum1 = sum1 + cm[i][j]
            else:
                sum2 = sum2 + cm[i][j]
                
total=sum1+sum2                
Accuracy=(sum1/total)*100            
print("The accuracy for the given test set is " + str(float(Accuracy)) + "%")

