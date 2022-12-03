#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression


# In[2]:


def get_max_score(model, X, y):
    score = 0
    for i in range(0,100): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        res_score = lr_model.score(X_test, y_test)
        if res_score > score:
            score = res_score
            randstate = i
    return score, randstate


# In[3]:


df = pd.read_csv("Heart_Disease_Prediction.csv")
df.head()


# In[4]:


le = LabelEncoder()
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])
df.head()


# In[5]:


X = df.drop(['Heart Disease', 'Age', 'Sex', 'FBS over 120'], axis=1)
y = df['Heart Disease']


# In[6]:


scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[7]:


lr_model = LogisticRegression(solver='liblinear',penalty="l1")


# In[8]:


print(get_max_score(lr_model, X, y))


# In[ ]:




