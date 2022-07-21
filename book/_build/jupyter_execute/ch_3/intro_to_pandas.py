#!/usr/bin/env python
# coding: utf-8

# # Introduction to Pandas

# The python package pandas is very useful to read csv files, but also many text files that are more or less formated as one observation per row and one column for each feature.

# As an example, we are going to look at the list of seismic stations from the Northern California seismic network, available here:
# 
# http://ncedc.org/ftp/pub/doc/NC.info/NC.channel.summary.day

# In[1]:


url = 'http://ncedc.org/ftp/pub/doc/NC.info/NC.channel.summary.day'


# First we import useful packages. The package request is useful to read data from a web page.

# In[2]:


import numpy as np
import pandas as pd
import io
import pickle
import requests
from datetime import datetime, timedelta
from math import cos, sin, pi, sqrt


# The function read_csv is used to open and read your text file. In the case of a well formatted csv file, only the name of the file needs to be entered:
# 
# data = pd.read_csv('my_file.csv')
# 
# However, many options are available if the file is not well formatted. See more on:
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

# In[3]:


s = requests.get(url).content
data = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None, skiprows=2, sep='\s+', usecols=list(range(0, 13)))
data.columns = ['station', 'network', 'channel', 'location', 'rate', 'start_time', 'end_time', 'latitude', 'longitude', 'elevation', 'depth', 'dip', 'azimuth']


# Let us look at the data. They are now stored into a pandas dataframe.

# In[4]:


data.head()


# There are two aways of looking at a particular column:

# In[5]:


data.station


# In[6]:


data['station']


# If we want to look at a given row or column, and we know its index, we can do:

# In[7]:


data.iloc[0]


# In[8]:


data.iloc[:, 0]


# If we know the name of the column, we can do:

# In[9]:


data.loc[:, 'station']


# We can also access a single value within a column:

# In[10]:


data.loc[0, 'station']


# We can filter the data with the value taken by a given column:

# In[11]:


data.loc[data.station=='KCPB']


# In[12]:


data.loc[(data.station=='KCPB') | (data.station=='KHBB')]


# In[13]:


data.loc[data.station.isin(['KCPB', 'KHBB'])]


# We can access to a brief summary of the data:

# In[14]:


data.station.describe()


# In[15]:


data.elevation.describe()


# We can perform standard operations on the whole data set:

# In[16]:


data.mean()


# In the case of a categorical variable, we can get the list of possile values that this variable can take:

# In[17]:


data.channel.unique()


# and get the number of times that each value is taken:

# In[18]:


data.station.value_counts()


# There are several ways of doing an operation on all rows of a column. The first option is to use the map function.
# 
# If you are not familiar with lambda function in Python, look at:
# 
# https://realpython.com/python-lambda/

# In[19]:


data_elevation_mean = data.elevation.mean()
data.elevation.map(lambda p: p - data_elevation_mean)


# The second option is to use the apply function:

# In[20]:


def remean_elevation(row):
    row.elevation = row.elevation - data_elevation_mean
    return row
data.apply(remean_elevation, axis='columns')


# We can also carry out simple operations on coulumns, provided they make sense.

# In[21]:


data.network + ' - ' + data.station


# A useful feature is to group the rows depending on the value of a categorical variable, and then apply the same operation to all the groups. For instance, I want to know how many times each station appears in the file:

# In[22]:


data.groupby('station').station.count()


# Or I want to know what is the lowest and the highest elevation for each station:

# In[23]:


data.groupby('station').elevation.min()


# In[24]:


data.groupby('station').elevation.max()


# We can have access to the data type of each column:

# In[25]:


data.dtypes


# Here, pandas does not recognize the start_time and end_time columns as a datetime format, so we cannot use datetime operations on them. We first need to convert these columns into a datetime format:

# In[26]:


# Transform column into datetime format
startdate = pd.to_datetime(data['start_time'], format='%Y/%m/%d,%H:%M:%S')
data['start_time'] = startdate
# Avoid 'OutOfBoundsDatetime' error with year 3000
enddate = data['end_time'].str.replace('3000', '2025')
enddate = pd.to_datetime(enddate, format='%Y/%m/%d,%H:%M:%S')
data['end_time'] = enddate


# We can now look when each seismic station was installed:

# In[27]:


data.groupby('station').apply(lambda df: df.start_time.min())


# The agg function allows to carry out several operations to each group of rows:

# In[28]:


data.groupby(['station']).elevation.agg(['min', 'max'])


# In[29]:


data.groupby(['station']).agg({'start_time':lambda x: min(x), 'end_time':lambda x: max(x)})


# We can also make groups by selecting the values of two categorical variables:

# In[30]:


data.groupby(['station', 'channel']).agg({'start_time':lambda x: min(x), 'end_time':lambda x: max(x)})


# Previously, we just printed the output, but we can also store it in a new variable:

# In[31]:


data_grouped = data.groupby(['station', 'channel']).agg({'start_time':lambda x: min(x), 'end_time':lambda x: max(x)})


# In[32]:


data_grouped.head()


# When we select only some rows, the index is not automatically reset to start at 0. We can do it manually. Many functions in pandas have also an option to reset the index, and option to transform the dataframe in place, instead of saving the results in another variable.

# In[33]:


data_grouped.reset_index()


# It is also possible to sort the dataset by value.

# In[34]:


data_grouped.sort_values(by='start_time')


# We can apply the sorting to several columns:

# In[35]:


data_grouped.sort_values(by=['start_time', 'end_time'])


# A useful pandas function is the merge functions that allows you two merge two dataframes that have some columns in common, but have also different columns that you may want to compare with each other.
# 
# For example, I have two earthquake catalogs. The 2007-2009 was established using data from a temporary experiment, and the 2004-2011 was established using data from a permanent seismic network. I would like to know if some earthquakes are detected by a network, but not by the other.

# I will compare the catalogs between July 2007 and May 2009. There is a time delay of 10s between the detection time of one catalog compared to the other. I will also filter the catalogs to eleiminate false detections.

# In[36]:


tbegin = datetime(2007, 9, 25, 0, 0, 0)
tend = datetime(2009, 5, 14, 0, 0, 0)
dt = 10.0
thresh1 = 1.4
thresh2 = 1.9


# I first read the two catalogs, and apply the filtering:

# In[37]:


namefile = 'catalog_2007_2009.pkl'
df1 = pickle.load(open(namefile, 'rb'))
df1 = df1[['year', 'month', 'day', 'hour', 'minute', 'second', 'cc', 'nchannel']]
df1 = df1.astype({'year': int, 'month': int, 'day': int, 'hour': int, 'minute': int, 'second': float, 'cc': float, 'nchannel': int})
date = pd.to_datetime(df1.drop(columns=['cc', 'nchannel']))
df1['date'] = date
df1 = df1[(df1['date'] >= tbegin) & (df1['date'] <= tend)]
df1_filter = df1.loc[df1['cc'] * df1['nchannel'] >= thresh1]

namefile = 'catalog_2004_2011.pkl'
df2 = pickle.load(open(namefile, 'rb'))
df2 = df2[['year', 'month', 'day', 'hour', 'minute', 'second', 'cc', 'nchannel']]
df2 = df2.astype({'year': int, 'month': int, 'day': int, 'hour': int, 'minute': int, 'second': float, 'cc': float, 'nchannel': int})
date = pd.to_datetime(df2.drop(columns=['cc', 'nchannel']))
df2['date'] = date
df2['date'] = df2['date'] - timedelta(seconds=dt)
df2 = df2[(df2['date'] >= tbegin) & (df2['date'] <= tend)]
df2_filter = df2.loc[df2['cc'] * df2['nchannel'] >= thresh2]


# To make the comparison, I first concatenate the two dataframes into a single dataframe. Then I merge the concatenated dataframe with one of the initial dataframes.
# 
# I apply the merge operation on the date column, that is if an earthquake in dataset 1 has the same date as an earthquake in dataset 2, I assume it is the same earthquake. You could also check if several columns have the same value, instead of doing the merge operation on only one column.
# 
# The process adds a merge column to the dataset, which indicates whether a row was found only in dataset 1, only in dataset 2, or in both datasets.

# In[38]:


# Earthquakes in filtered 2007-2009 catalog but not in (unfiltered) 2004-2011 catalog
df_all = pd.concat([df2, df1_filter], ignore_index=True)
df_merge = df_all.merge(df2.drop_duplicates(), on=['date'], how='left', indicator=True)
df_added_1 = df_merge[df_merge['_merge'] == 'left_only']

# Earthquakes in filtered 2004-2011 catalog but not in (unfiltered) 2007-2009 catalog
df_all = pd.concat([df1, df2_filter], ignore_index=True)
df_merge = df_all.merge(df1.drop_duplicates(), on=['date'], how='left', indicator=True)
df_added_2 = df_merge[df_merge['_merge'] == 'left_only']


# In[39]:


df_added_1


# In[40]:


df_added_2


# In[ ]:




