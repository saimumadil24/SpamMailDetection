#!/usr/bin/env python
# coding: utf-8

# In[180]:


#Importing Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import pickle as pk


# In[181]:


#Importing the dataset
data=pd.read_csv('emails.csv')
data


# In[182]:


#Checking the duplicates cells
data.duplicated().sum()


# In[183]:


#Dropping the duplicates cells
data=data.drop_duplicates()


# In[184]:


#Again chedcking for duplicates
data.duplicated().sum()
data


# In[185]:


#Taking Independend and Dependent Variable
x=data['text']
y=data['spam']


# In[186]:


#Calling the scikit libraries
cv=CountVectorizer()
mnb=MultinomialNB()


# In[187]:


#Split the dataset into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)


# In[188]:


#Transforming the string into integer
x_train=cv.fit_transform(x_train)


# In[189]:


#Fitting the values in scikitlern lbrary
mnb.fit(x_train,y_train)


# In[190]:


#Getting ready for testing the test part
x_test=cv.transform(x_test)


# In[191]:


#Prediction for the test set
pred=mnb.predict(x_test)
pred


# In[192]:


#Checking the accuracy score
mnb.score(x_test,y_test)


# In[193]:


#Checking two mails! 1=Spam, 0= Ham
mails=['Congratulations, you have won an iphone 14 pro max. Plz, Click the link below. And provide the information that we need','Hey, I am Saimum Adil Khan from Kaggle. I am an expert data scientist. I have a Job offer for you']
mails=cv.transform(mails)


# In[194]:


#Seeing the array of those text
mails.toarray()


# In[195]:


#Finally get the prediction
mnb.predict(mails)


# In[196]:


with open('spam_mail.pkl','wb') as f:
    pk.dump((mnb,cv),f)

