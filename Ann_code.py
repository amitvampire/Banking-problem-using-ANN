import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Churn_Modelling.csv')

X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder1=LabelEncoder()
X[:,1]=labelencoder1.fit_transform(X[:,1])
labelencoder2=LabelEncoder()
X[:,2]=labelencoder2.fit_transform(X[:,2])
onehot=OneHotEncoder(categorical_features=[1])
X=onehot.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y ,test_size=.20 ,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
#initilizing ANN model
classifier = Sequential()
#creating input layer and first hidden layer
classifier.add(Dense(units=6,input_shape=(11,),activation = 'relu' , kernel_initializer = 'uniform' ))
# here uniform means initializing values of W(weights ) close to 0 but not 0
#creating second hidden layer 
classifier.add(Dense(units=6,activation='relu', kernel_initializer = 'uniform'))

#creating output layer
classifier.add(Dense(units=1,activation = 'sigmoid' , kernel_initializer = 'uniform' )) 

#compiling classifier 
classifier.compile(optimizer = 'adam' , loss='binary_crossentropy',metrics=['accuracy'])

#fitting 
classifier.fit(X_train,y_train,batch_size=10,epochs=10) 

#predicting 
y_pred = classifier.predict(X_test)

y_pred =(y_pred > .5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#printing accuracy on test set
print( (cm[0:1,0:1]+cm[1:2,1:2])/(2000) )

#checking on a particular data

new_pred=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred=(new_pred>.5)
print(new_pred)










