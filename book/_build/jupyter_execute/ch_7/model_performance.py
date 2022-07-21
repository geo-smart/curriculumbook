#!/usr/bin/env python
# coding: utf-8

# # Model Performance
# 
# 
# 
# To measure a fit performance, we will measure several metrics.
# 
# 1. Standard Error
# 
# The variance of tge errors is reduced by fitting the data (scatter points) against the model. The variance is:
# 
# $\text{Var}(x) = 1/n \sum_{i=1}^n (x_i-\hat{x})^2$, 
# 
# where $\hat{x}$ is the mean of $x$.  When fitting the regression, we predict the values $x_{pred}$. The residuals are the differences between the data and the predicted values: $e = x - x_{pred} $. $R^2$ or <i> coefficient of determination</i> is:
# 
# $R^2 = 1 - \text{Var}(x-x_{pred}) /\text{Var}(x)  = 1 - \text{Var}(e) /\text{Var}(x) $
# The smaller the error, the "better" the fit (we will discuss later that a fit can be too good!), the closter $R^2$ is to one.
# 
# 2. t-statistics
# 
# 
# 3. p-value
# 
# 4. F-statistics

# In[1]:


import requests, zipfile, io, gzip, glob, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

sta="P395"
file_url="http://geodesy.unr.edu/gps_timeseries/tenv/IGS14/"+ sta + ".tenv"
r = requests.get(file_url).text.splitlines()  # download, read text, split lines into a list
ue=[];un=[];uv=[];se=[];sn=[];sv=[];date=[];date_year=[];df=[]
for iday in r:  # this loops through the days of data
    crap=iday.split()
    if len(crap)<10:
      continue
    date.append((crap[1]))
    date_year.append(float(crap[2]))
    ue.append(float(crap[6])*1000)
    un.append(float(crap[7])*1000)
    uv.append(float(crap[8])*1000)
#             # errors
    se.append(float(crap[10])*1000)
    sn.append(float(crap[11])*1000)
    sv.append(float(crap[12])*1000)

  # make dataframe
crap={'station':sta,'date':date,'date_year':date_year,'east':ue,'north':un,'up':uv}
if len(df)==0:
    df = pd.DataFrame(crap, columns = ['station', 'date','date_year','east','north','up'])
else:
    df=pd.concat([df,pd.DataFrame(crap, columns = ['station', 'date','date_year','east','north','up'])])
df.describe()



# In[ ]:


# now let's find the trends and detrend the data.
from scipy import stats
# linear regression such that: displacement = Velocity * time
# velocity in the East componentone.
Ve, intercept, r_value, p_value, std_err = stats.linregress(df['date_year'][df['station']==sta],df['east'][df['station']==sta])
# horizontal plate motion:
print(sta,"overall plate motion there",Ve,'mm/year')
print("parameters: Coefficient of determination %f4.2, P-value %f4.2, standard deviation of errors %f4.2"\
      %(r_value,p_value,std_err))
from sklearn.linear_model import LinearRegression
# convert the data into numpy arrays.
E = np.asarray(df['east'][df['station']==sta]).reshape(-1, 1)# reshaping was necessary to be an argument of Linear regress
t = np.asarray(df['date_year'][df['station']==sta]).reshape(-1, 1)
tt = np.linspace(np.min(t),np.max(t),1000)

# perform the linear regression. First we will use the entire available data
regr = LinearRegression()
# we will first perform the fit:
regr.fit(t,E)
# We will first predict the fit:
Epred=regr.predict(t) 

# The coefficients
print('Coefficient / Velocity eastward (mm/year): ', regr.coef_[0][0])

plt.plot(t,E);ax[0].grid(True);ax[0].set_ylabel('East (mm)')
plt.plot(t,Epred,color="red")
plt.grid(True)
plt.xticks(())
plt.yticks(())
plt.show()

