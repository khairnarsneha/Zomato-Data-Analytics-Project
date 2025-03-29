#!/usr/bin/env python
# coding: utf-8

# # Zomato data analysis project

# # Step 1 - Importing Libraries
pandas is used for data manipulation and analysis.
numpy is used for numerical operations.
matplotlib.pyplot and seaborn are used for data visualization.
# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# # Step 2 - Create the data frame

# In[5]:


import pandas as pd

dataframe = pd.read_csv("Zomato data .csv")
print(dataframe)


# In[6]:


dataframe


# # Convert the data type of column - rate

# In[7]:


def handleRate(value):
    value = str(value).split('/')
    value = value[0];
    return float(value)

dataframe['rate']=dataframe['rate'].apply(handleRate)
print(dataframe.head())


# In[8]:


dataframe.info()


# # Type of resturant

# In[9]:


dataframe.head()


# # Question - What type of restaurant do majority of customers order from?

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=dataframe['listed_in(type)'])
plt.xlabel("type of resturant")


# # conclusion - majority of the restaurant falls in dinning category

# # Question - How many votes has each type of restaurant received from customers?

# In[11]:


dataframe.head()


# In[18]:


grouped_data = dataframe.groupby('listed_in(type)')['votes'].sum()
result = pd.DataFrame({'votes': grouped_data})
plt.plot(result, c="green", marker="o")
plt.xlabel("Type of restaurant",c="Blue", size=20)
plt.ylabel("Votes", c="red", size=20)


# # conclusion - dinning restaurant has received maximum votes

# # Question - What are the ratings that the majority restaurants have received?

# In[19]:


dataframe.head()


# In[21]:


plt.hist(dataframe['rate'],bins = 10)
plt.title("ratings distribution")
plt.show()


# # conclusion - the majority restaurants received ratings from 3.5 to 4

# # Question - Zomato has observed that most couples order most of their food online.what is their average spending on each order?

# In[22]:


dataframe.head()


# In[23]:


couple_data=dataframe['approx_cost(for two people)']
sns.countplot(x=couple_data)


# # conclusion - the majority of couples prefer restaurants with an approximate cost of 300 rupees

# # Question - Which mode (online or offline) has received the maximum rating?

# In[24]:


dataframe.head()


# In[25]:


plt.figure(figsize = (6,6))
sns.boxplot(x = 'online_order', y = 'rate', data = dataframe)


# # conclusion - offline order received lower rating in comparison to online order

# # Question -  which type restaurant received more offline orders, so that Zomato can prove customers with some good offers?

# In[26]:


dataframe.head()


# In[28]:


pivot_table = dataframe.pivot_table(index = 'listed_in(type)', columns = 'online_order', aggfunc = 'size', fill_value = 0)
sns.heatmap(pivot_table, annot = True, cmap = "YlGnBu", fmt = 'd')
plt.title("Heatmap")
plt.xlabel("Online Order")
plt.ylabel("Listed In (Type)")
plt.show()


# # conclusion - Dinning restaurants primarily accept offline orders, whereas cafes primarily receive online orders. This suggests that clients prefered oreders in person at restaurants, but prefer online ordering at cafes.

# In[ ]:




