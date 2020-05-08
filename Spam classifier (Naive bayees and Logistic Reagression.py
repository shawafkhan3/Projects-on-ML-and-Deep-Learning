#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
data=pd.read_csv("C:\\Users\\Shawaf Khan\\Desktop\\spam.csv",encoding="ISO-8859-1")
data["v1"].replace(to_replace=["ham","spam"],value=[0,1],inplace=True)
data.shape


# In[26]:


data.drop_duplicates(inplace=True)


# In[27]:


data.shape


# In[28]:


data.columns


# In[33]:


data.isnull().sum()


# In[31]:


import numpy as np
import nltk
from nltk.corpus import stopwords
import string


# In[38]:


def process(text):
    new=[char for char in text if char not in string.punctuation]
    new=''.join(new)
    clean=[ch for ch in new.split() if ch not in stopwords.words('english') ]
    return clean
    


# In[40]:


##To check weather the function is working
data["v2"].head().apply(process)


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer 
skt=CountVectorizer(analyzer=process).fit_transform(data["v2"])


# In[46]:


### creating training data and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(skt,data["v1"],test_size=0.20,random_state=0)


# In[47]:


skt.shape


# In[49]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(x_train,y_train)


# In[51]:


print(classifier.predict(x_train))
print(y_train.values)


# In[62]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
predict=classifier.predict(x_train)
print("Classification Report:   ",classification_report(y_train,predict))
print("Confusion Matrix:   \n",confusion_matrix(y_train,predict))
print("Acurracy:  ",accuracy_score(y_train,predict))


# In[61]:


### REport for test data
predict_test=classifier.predict(x_test)
print("Classification Report:   ",classification_report(y_test,predict_test))
print("Confusion Matrix:  \n ",confusion_matrix(y_test,predict_test))
print("Acurracy:  ",accuracy_score(y_test,predict_test))


# In[64]:


print("predicted value:   ",classifier.predict(x_test))
print("Actual values:   ",y_test.values)


# In[66]:


# Creating spam classifier using Logistic regression


# In[67]:


x_train1,x_test1,y_train1,y_test=train_test_split(skt,data["v1"],test_size=0.30,random_state=20)


# In[71]:


from sklearn.linear_model import LogisticRegression
classifierr=LogisticRegression(solver='liblinear',penalty='l2')
classifierr.fit(x_train1,y_train1)


# In[76]:


pred=classifierr.predict(x_train1)

print(accuracy_score(y_train1,pred))


# In[ ]:




