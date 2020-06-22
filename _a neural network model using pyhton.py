#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


data=datasets.load_digits()


# In[23]:


print(data.images)


# In[24]:


print(data.target)


# In[25]:


Y1=data.target
print(Y1.shape)
n=len(data.images)
X=data.images.reshape(n,-1)
Y1=data.target.reshape(1797,1)
print(X.shape)
print(Y1.shape)


# In[26]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y1,test_size=0.20,random_state=0)


# In[ ]:





# In[27]:


index = 25
plt.imshow(data.images[index])
print ("The value of y in this picture is = " + str(data.target[index]))


# In[ ]:





# In[28]:


print(X_train.shape)
print(data.images.shape)


# In[29]:


X_train=X_train/255
X_test=X_test/255


# In[30]:


print(X_train)


# In[31]:


def sigmoid(x):
    a=1/(1+(np.exp(-x)))
    return a


# In[32]:


def withzero(dim):
    w=np.zeros([dim,1])
    b=0
    
    assert(w.shape==(dim,1))
    assert(isinstance(b,float)or isinstance(b,int))
    
    return w,b


# In[33]:


## Just to check the above function nothing to do with the code
dim = 2
w, b = withzero(dim)
print ("w = " + str(w))
print ("b = " + str(b))


# In[34]:


def propagate(w,b,X,Y):
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    
    cost=(-1/m)*np.sum((Y*np.log(A)+(1-Y)*np.log(1-A)))
    
    dw=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y)
    
    
    assert(db.dtype==float)
    cost=np.squeeze(cost)
    assert(cost.shape==())
    grads ={"dw":dw,"db":db}
    
    return grads,cost
    
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


def optimise(w,b,X,Y,num_iterations,learning_rate,print_cost=True):
    costs=[]
    for i in range(num_iterations):
        
        gards,cost=propagate(w,b,X,Y)
        dw=gards["dw"]
        db=gards["db"]
        
        w=w-(learning_rate*dw)
        b=b-(learning_rate*db)
        
        if (i%100==0):
            costs.append(cost)
            
        if print_cost and i%100==0:
            print("Cost function at iteration {}, {}".format(i,cost))
            
    params={"w":w,"b":b}
    gards={"dw":dw,"db":db}
    
    return params,grads,costs
            
            
            
        
            
    
    
            


# In[ ]:





# In[36]:


def predict(w,b,X):
    m=X.shape[1]
    Y_predict=np.zeros([1,m])
    w=w.reshape(X.shape[0],1)
    
    A=sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        if(A[0,i]>=0.5):
            Y_predict[0,i]=1
        elif(A[0,i]<0.5):
            Y_predict[0,i]=0
    assert(Y_predict.shape == (1, m))        
    return Y_predict


# In[37]:


def model(X_train1, Y_train1, X_test1, Y_test1, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w,b=withzero(X_train1.shape[0])
    
    parameters,grads,cost=optimise(w, b, X_train1, Y_train1, num_iterations, learning_rate, print_cost)
    
    w=parameters["w"]
    b=parameters["b"]
    
    Y_training=predict(w,b,X_train1)
    Y_testing=predict(w,b,X_test1)
    
    print("Train accuracy is ==",100-np.mean(np.abs(Y_training-Y_train1)*100))
    print("Train accuracy is ==",100-np.mean(np.abs(Y_testing-Y_test1)*100))
    
    
    d={"cost":costs,"Y_prediction training":Y_training,"Y_prediction test":Y_testing,"w":w,"b":b,"Learning Rate":learning_rate,"Iterations":num_iterations} 
    return d


# In[ ]:


d=model(X_train,Y_train,X_test,Y_test,num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# In[ ]:




