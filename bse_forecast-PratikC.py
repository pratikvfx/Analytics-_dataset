
# coding: utf-8

# In[4]:


import warnings
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import pyramid as pm
from sklearn import metrics
from pyramid.arima import auto_arima
warnings.filterwarnings('ignore')


# In[5]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactively ="all"


# In[6]:


bse=pd.read_csv('C:\\Users\\Shriyans\\Desktop\\bse.csv')


# In[7]:


bse.head()


# In[8]:


bse.describe()


# In[9]:


day =pd.date_range('20171027', periods=248, freq='D')
day


# In[10]:


bse['datestamp'] = day
bse.head()


# In[11]:


data = bse.loc[:,('datestamp', 'Close')]
data.head()


# In[12]:


data.describe()


# In[13]:


data.set_index('datestamp', inplace=True)
data.head()


# In[14]:


plt.figure(figsize=(15,10))
plt.plot(data.Close)
plt.xlabel('Time')
plt.ylabel('Sensex Point')
plt.title('# Sensex')
plt.show;


# In[34]:


decomposition = seasonal_decompose(data, model ='additive')


# In[35]:


decomposition


# In[31]:


plt.figure(figsize=(15,10))

trend = decomposition.trend
seasonal=decomposition.seasonal
residual = decomposition.resid

plt.subplot(221)
plt.plot(data,'r', label='Original')
plt.legend(loc='best')
plt.subplot(222)
plt.plot(trend, 'b', label='Trend')
plt.legend(loc='best')
plt.subplot(223)
plt.plot(seasonal,'g', label='Seasonality')
plt.legend(loc='best')
plt.subplot(224)
plt.plot(residual, 'y', label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show();


# In[36]:


from pyramid.arima.stationarity import ADFTest
adf_test = ADFTest(alpha=0.05)
adf_test.is_stationary(data)


# In[37]:


train, test = data[:200], data[200:]
test.shape


# In[38]:


train.shape


# In[39]:


plt.plot(train)
plt.plot(test)
plt.show();


# In[43]:


from pyramid.arima import auto_arima

Arima_model = auto_arima(train, start_p=1, max_p=8, max_q=8, start_P=0, start_Q=0, max_P=8, max_Q=8, m=1,
                        seasonal=True, trace=True, d=1, D=1, error_action='warn', suppress_warnings=True,
                        stepwise=True, random_state=20, n_fits=30)
Arima_model.summary()


# In[44]:


prediction = pd.DataFrame(Arima_model.predict(n_periods=48), index=test.index)
prediction.columns=['Predicted_SensexPoint']


# In[45]:


prediction


# In[46]:


plt.figure(figsize=(15,10))
plt.plot(train, label='Training')
plt.plot(test,label='Test')
plt.plot(prediction, label='Predicted')
plt.legend(loc='upper center')
plt.show();


# In[47]:


test['Predicted_Points'] = prediction
test['Error'] = test['Close'] - test['Predicted_Points']
test


# In[48]:


metrics.mean_absolute_error(test.Close, test.Predicted_Points)


# In[50]:


metrics.mean_squared_error(test.Close, test.Predicted_Points)


# In[51]:


metrics.median_absolute_error(test.Close, test.Predicted_Points)


# In[52]:


plt.figure(figsize=(20,10))
plt.subplot(121)
plt.plot(test.Error, color='#ff33CC')
plt.subplot(122)
scipy.stats.probplot(test.Error, plot=plt)
plt.show;


# In[53]:


plt.figure(figsize=(20,10))
pm.autocorr_plot(test.Error)
plt.show();

