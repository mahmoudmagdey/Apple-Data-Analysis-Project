#!/usr/bin/env python
# coding: utf-8

# # Import the dependecies

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display plots inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Import Prophet for time series forecasting
try:
    from prophet import Prophet
except Exception:
    from fbprophet import Prophet


# # Import the data

# In[3]:


Apple_Sales = pd.read_csv('Final_Project.csv', encoding='latin1')


# # Explore the Data

# In[4]:


# Get the First 5 rows
Apple_Sales.head()


# In[5]:


# get Information About the data to check null and duplicated values
Apple_Sales.info()


# In[6]:


# Sort The Data According to Sale Date
Apple_Sales = Apple_Sales.sort_values('Sale_Date')
Apple_Sales.head()


# In[7]:


# Make New Column [Revenue] to predict the Total sales
Apple_Sales['Revenue'] = Apple_Sales['Quantity'] * Apple_Sales['Product_Price']
Apple_Sales.head()


# In[20]:


# Convert The Type of Sale_Date Column to Date data Type
Apple_Sales['Sale_Date'] = pd.to_datetime(Apple_Sales['Sale_Date'])


# In[21]:


# Group by Sale_Date, sum the Revenue for each day
daily_revenue = Apple_Sales.groupby('Sale_Date')['Revenue'].sum().reset_index()


# In[22]:


# Plot daily revenue
plt.figure(figsize=(18,8))
plt.plot(daily_revenue['Sale_Date'], daily_revenue['Revenue'])

plt.xlabel("Sale Date")
plt.ylabel("Total Revenue")
plt.title("Daily Revenue Over Time")
plt.xticks(rotation=45)
plt.show()


#  # Prepare data for Prophet

# In[23]:


Apple_Sales_ts = Apple_Sales[['Sale_Date','Revenue']]
Apple_Sales_ts


# In[26]:


# Prophet requires columns named 'ds' for date and 'y' for values
Apple_Sales_ts = daily_revenue.rename(columns={'Sale_Date':'ds', 'Revenue':'y'})


# In[32]:


# Initialize Prophet model
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)


#  # Fit the model

# In[33]:


# Train the model on historical daily revenue
m.fit(Apple_Sales_ts)


# In[34]:


# Predict 365 days into the future
future = m.make_future_dataframe(periods=365)


# In[35]:


# Generate predictions
forecast = m.predict(future)


# In[36]:


# Plot the forecast
fig1 = m.plot(forecast)
plt.title("Revenue Forecast Using Prophet")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()


# In[ ]:




