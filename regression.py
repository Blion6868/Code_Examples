#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\bryan\Desktop\New_Data_Science_Class\08-Linear-Regression-Models\Advertising.csv")


# In[3]:


df.head()


# In[4]:


df.corr()


# In[5]:


df['total_spent'] = df['TV'] + df['radio'] + df['newspaper']


# In[6]:


df.head()


# In[9]:


sns.scatterplot(data=df, x='total_spent',y='sales')


# In[10]:


sns.regplot(data=df, x='total_spent',y='sales')


# In[43]:


X1 = df['total_spent']
y1 = df['sales']


# In[44]:


#y = mx+b
#y = B1x + B0

#coefficients
np.polyfit(X1, y1, deg=1)


# In[45]:


potential_spend = np.linspace(0,500,100)


# In[46]:


predicted_sales = 0.04868788 * potential_spend + 4.24302822


# In[47]:


sns.scatterplot(x='total_spent',y='sales', data=df)
plt.plot(potential_spend,predicted_sales, color='red')


# In[48]:


spend = 200

predicted_sales = 0.04868788 * spend + 4.24302822


# In[49]:


predicted_sales


# In[50]:


np.polyfit(X1,y1,3)


# In[51]:


pot_spend = np.linspace(0,500,100)


# In[31]:


pred_sales = 3.07615033e-07*pot_spend**3 +  -1.89392449e-04*pot_spend**2 +  8.20886302e-02*pot_spend +  2.70495053e+00


# In[32]:


pred_sales


# In[35]:


sns.scatterplot(x='total_spent',y='sales', data=df)
plt.plot(pot_spend,pred_sales, color='red')


# In[36]:


df.head()


# In[262]:


df = df.drop('total_spent',axis=1)


# In[263]:


df.head()


# In[264]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

axes[0].plot(df['TV'],df['sales'],'o')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].set_ylabel("Sales")
axes[1].set_title("Radio Spend")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].set_ylabel("Sales")
axes[2].set_title("Newspaper Spend")
plt.tight_layout();


# In[265]:


sns.pairplot(df)


# In[266]:


X = df.drop('sales',axis=1)


# In[267]:


y = df['sales']


# In[268]:


from sklearn.model_selection import train_test_split


# In[269]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[270]:


len(X_train)


# In[271]:


len(X_test)


# In[272]:


len(y_train)


# In[273]:


len(y_test)


# In[274]:


from sklearn.linear_model import LinearRegression


# In[275]:


model = LinearRegression()


# In[276]:


model.fit(X_train, y_train)


# In[277]:


pred = model.predict(X_test)


# In[278]:


from sklearn.metrics import mean_absolute_error

#mean of the absolute value of errors

mean_absolute_error(y_test, pred)


# In[279]:


from sklearn.metrics import mean_squared_error

#larger errors punished more than MAE

mean_squared_error(y_test, pred)


# In[280]:


#RMSE

np.sqrt(mean_squared_error(y_test, pred))


# In[281]:


sns.histplot(data=df, x='sales',bins=20)


# In[282]:


df['sales'].mean()


# In[283]:


test_residuals = y_test - pred


# In[284]:


test_residuals


# In[285]:


sns.scatterplot(x=pred, y=test_residuals)
plt.axhline(y=0, color='r',ls='dashed')


# In[286]:


sns.distplot(test_residuals, bins=25,kde=True)


# In[287]:


import scipy as sp


# In[288]:


#create a figure and axis to plot on
fig, ax = plt.subplots(figsize=(6,8),dpi=100)

#probplot returns the raw values if needed
_ = sp.stats.probplot(test_residuals, plot=ax)


# In[289]:


final_model = LinearRegression()


# In[290]:


final_model.fit(X,y)


# In[291]:


final_model.coef_


# In[300]:


final_model.intercept_


# In[297]:


y_hat = final_model.predict(X)


# In[299]:


fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

axes[0].plot(df['TV'],df['sales'],'o')
axes[0].plot(df['TV'],y_hat,'o',color='red')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].plot(df['radio'],y_hat,'o',color='red')
axes[1].set_title("Radio Spend")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].plot(df['radio'],y_hat,'o',color='red')
axes[2].set_title("Newspaper Spend");
axes[2].set_ylabel("Sales")
plt.tight_layout();


# In[304]:


from joblib import dump, load


# In[306]:


dump(final_model,'final_sales_model.joblib')


# In[307]:


loaded_model = load('final_sales_model.joblib')


# In[308]:


loaded_model.coef_


# In[309]:


campaign = [[149,22,12]]


# In[310]:


loaded_model.predict(campaign)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




