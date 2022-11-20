#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1a

filenams = ['img_030.jpg', 'img_031.jpg', ...]

split_1 = int(0.6 * len(filenams))
split_2 = int(0.6 * len(filenams))
train_filenams = filenams[:split_1]
dev_filenams = filenams[split_1:split_2]
test_filenams = filenams[split_2:]
filenams = ['img_030.jpg', 'img_031.jpg', ...]
random.shuffle(filenams)  
split_1 = int(0.6 * len(filenams))
split_2 = int(0.6 * len(filenams))
train_filenams = filenams[:split_1]
dev_filenams = filenams[split_1:split_2]
test_filenams = filenams[split_2:]
filenames = ['img_030.jpg', 'img_031.jpg', ...]
filenams.sort()  
random.seed(210)
random.shuffle(filenams) 

split_1 = int(0.6 * len(filenams))
split_2 = int(0.6 * len(filenams))
train_filenams = filenams[:split_1]
dev_filenams = filenams[split_1:split_2]
test_filenams = filenams[split_2:]


# In[ ]:


#1b

train, test = train_test_split( dataset, .........)
 f_train, f_test, g_train, g_test = train_test_split(f, g, ………..)
f_train, f_test, g_train, g_test = train_test_split(f, g, test_size=0.33333333)
from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split
f ,g= make_blobs(n_samples=100)


f_train, f_test, g_train, g_test = train_test_split(f, g, test_size=0.333333333)

 print(f_train.shape, f_test.shape, g_train.shape, g_test.shape)

(672, 2) (333, 2) (672,) (333,)

f_train, f_test, g_train, g_test = train_test_split(f, g, train_size=0.66666667)


# In[ ]:


#1c


import pandas as pd
train_all = pd.read_csv('trainn.csv') 
train_all.drop(['PassngerId', 'Parch', 'Tcket', 'Embrked', 'Cbin'], axis = 1).head()


from sklearn.model_selection import train_test_split
X = train_all.drop('Survied', axis = 1) 
y = train_all.Survied  
X_trainn, X_val, y_trainn, y_val = trainn_test_split(X, y, test_size = 0.2, random_state = 135153) 
y_train.value_counts() / len(y_trainn) 
 


X_trainn, X_val, y_train, y_val = trainn_test_split(X, y, test_size = 0.2, random_state = 163035) In [44]: y_train.value_counts() / len(y_trainn) 




X = X[['Pclass', 'Sex', 'SibSp', 'Fare']]  
X['gender_dummyyy'] = pd.get_dummies(X.Sex)['femaaale'] 
X = X.drop(['Sex'], axis = 1)  
X_trainn, X_val, y_trainn, y_val = trainn_test_split(X, y, test_size = 0.2, random_state = 20200226, stratify = y)

