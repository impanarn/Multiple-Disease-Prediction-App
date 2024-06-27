# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""***Reading The Dataset***"""

heart=pd.read_csv('C:/Users/Impana/OneDrive/Desktop/vital_forecast/heart.csv')
heart.head()

"""***Checking the shape of DataFrame***"""

print('Number of rows are',heart.shape[0], 'and number of columns are ',heart.shape[1])

"""***Checking for null values***"""

heart.isnull().sum()/len(heart)*100

"""**No null values found**

***Checking For datatypes of the attributes***
"""

heart.info()

"""**All attributes are of type 'int' except 'oldpeak'**

***Checking for duplicate rows***
"""

heart[heart.duplicated()]

"""***Removing the duplicates***"""

heart.drop_duplicates(keep='first',inplace=True)

"""**Checking new shape**"""

print('Number of rows are',heart.shape[0], 'and number of columns are ',heart.shape[1])

"""***Checking statistical data***"""

heart.describe()

heart['output'].value_counts()

"""***Computing the correlation matrix***"""

heart.corr()


"""# **Data preprocessing**"""

x = heart.iloc[:, :-1].values
y = heart.iloc[:, -1].values
x,y

"""**Splitting the dataset into training and testing data**"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

print('Shape for training data', x_train.shape, y_train.shape)
print('Shape for testing data', x_test.shape, y_test.shape)

"""**Feature Scaling**"""

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train,x_test

"""**Support Vector Machine**"""

model = SVC()
model.fit(x_train, y_train)

predicted = model.predict(x_test)
print("The accuracy of SVM is : ", accuracy_score(y_test, predicted)*100, "%")

model = SVC()
model.fit(x_train, y_train)

predicted1 = model.predict(x_train)
print("The accuracy of SVM is : ", accuracy_score(y_train, predicted1)*100, "%")

input_data = (51,0,2,130,256,0,0,149,0,0.5,2,0,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

input_data_scaled = scaler.transform(input_data_reshaped)

# Make prediction
prediction = model.predict(input_data_scaled)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

import pickle

filename = 'heart_trained_model.sav'
pickle.dump((model, scaler), open(filename, 'wb'))

# loading the saved model
loaded_model, loaded_scaler = pickle.load(open('C:/Users/Impana/Downloads/VitalForecast/heart_trained_model.sav', 'rb'))

input_data = (54,1,0,122,286,0,0,116,1,3.2,1,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

input_data_scaled = scaler.transform(input_data_reshaped)

# Make prediction
prediction = loaded_model.predict(input_data_scaled)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

