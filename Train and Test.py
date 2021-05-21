#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
df = pd.read_csv('C:/Users/Abhi/Desktop/data/Data of car.csv')
df.head()


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


####Car Mileage Vs Sell Price ($)


# In[20]:


plt.scatter(df['Age(Years)'], df['Sell Price($)'])


# In[21]:


plt.scatter(df['Mileage'], df['Sell Price($)'])


# In[22]:


x = df[['Mileage', 'Age(Years)']]
y = df['Sell Price($)']


# In[23]:


x


# In[24]:


y


# In[67]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[68]:


len(y_test)


# In[46]:


df.shape


# In[69]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train,y_train)


# In[70]:


clf.predict(x_test)


# In[71]:


y_test


# In[72]:


clf.score(x_test,y_test)


# In[ ]:




