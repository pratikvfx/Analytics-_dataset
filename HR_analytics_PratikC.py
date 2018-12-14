
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


df=pd.read_csv("/Users/akanksha/Documents/study/Reva/Worksheet in 2_HR analytics.csv", sep='|')


# In[6]:


df


# In[7]:


df.info()


# In[8]:


df.corr()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


#finding total employee count
number_of_employees =  df.shape[0]+1
print('Number of employees = ', number_of_employees)


# In[13]:


#finding % of employees who left vs stayed back
employees_who_left = len(df[df['If employee has left'] == 1])
print('Number of employees who have left = ', employees_who_left)
print('Percentage of employees who have left = %0.2f' % (employees_who_left/15000 * 100))


# In[16]:


fig1 = plt.figure(figsize=(14,6))
labels = ['Stayed', 'Left']


plt.pie(df['If employee has left'].value_counts(), explode=[0,0.1], labels=labels,autopct='%1.1f%%', shadow=True)


# In[18]:


#the no.of employees left belonged to any specific salary bracket or department.

from sklearn import preprocessing

le_dept = preprocessing.LabelEncoder()
le_dept.fit(df['department'])
df['dept'] = le_dept.transform(df['department'])

le_salary = preprocessing.LabelEncoder()
le_salary.fit(df['salary bracket'])

df['salary'] = le_salary.transform(df['salary bracket'])

df_x = df.drop(columns=['department', 'salary bracket'])
df_x.head()


# In[21]:


fig2 = plt.figure()
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig2=plt.gcf()
fig2.set_size_inches(10,5)


# In[23]:


from matplotlib import gridspec

fig3 = plt.figure(figsize=(20,6))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3]) 

sns.countplot('salary bracket', hue='If employee has left', data=df)


# In[24]:


sns.countplot('department', hue='If employee has left', data=df)


# In[26]:


#low satisfaction level maybe?

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)
sns.set_style('darkgrid')
fig4 = plt.figure(figsize=(10,14))
ax1 = fig4.add_subplot(2,1,1)
ax1 = sns.boxplot(x='salary bracket', y="satisfaction_level", hue="If employee has left", data=df, palette="BrBG")
ax1.legend(loc=(1.1, 0.5), title='If employee has left')


# In[30]:


#long working hours maybe?

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)
sns.set_style('darkgrid')
fig4 = plt.figure(figsize=(10,14))
ax1 = fig4.add_subplot(2,1,1)
ax1 = sns.boxplot(x='salary bracket', y="average_montly_hours", hue="If employee has left", data=df, palette="BrBG")
ax1.legend(loc=(1.1, 0.5), title='If employee has left')

