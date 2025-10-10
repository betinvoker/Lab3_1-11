import pandas as pd
import numpy as np
import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from tensorflow.keras.models import Sequential
from keras.layers import Dense
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 

print('\n', X)
print('\n', y)

y.loc[y['Diagnosis'] == 'M']=0.
y.loc[y['Diagnosis'] == 'B']=1.

print('\n', y)

X_train = X.sample(frac=0.8, random_state=0)
X_test = X.drop(X_train.index)

Y_train = y.sample(frac=0.8, random_state=0)
Y_test = y.drop(Y_train.index)

print('\n', Y_test)

X_train = np.array(X_train).astype(np.float32)
Y_train = np.array(Y_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
Y_test = np.array(Y_test).astype(np.float32)

print('\n', Y_test)

classifier = Sequential() # Инициализация НС
classifier.add(Dense(units=16, activation='relu', input_dim=30))
classifier.add(Dense(units=8, activation='relu'))
classifier.add(Dense(units=6, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy')
classifier.fit(X_train, Y_train, batch_size=1, epochs=100)

Y_pred = classifier.predict(X_test)
Y_pred = [1 if y >= 0.5 else 0 for y in Y_pred]

total= 0
correct = 0
wrong = 0
for i in range(len(Y_pred)):
    total = total + 1
    if (Y_test[i] == Y_pred[i]):
        correct = correct + 1
    else:
        wrong = wrong + 1

print(f'Total {str(total)}\nCorrect {str(correct)}\nWrong {str(wrong)}')
print('\n', X_test[10], '\n', Y_test[10], '\n', Y_pred[10], '\n')

Y_pred = classifier.predict(X_test)
Y_pred = [1 if y >= 0.5 else 1 for y in Y_pred]

total= 0
correct = 0
wrong = 0
for i in range(len(Y_pred)):
    total = total + 1
    if (Y_test[i] == Y_pred[i]):
        correct = correct + 1
    else:
        wrong = wrong + 1

print(f'Total {str(total)}\nCorrect {str(correct)}\nWrong {str(wrong)}')
print('\n', X_test[10], '\n', Y_test[10], '\n', Y_pred[10])

tf.test.gpu_device_name()