#!/usr/bin/env python
# coding: utf-8

# # Databases
# This tutorial will cover the basics of building a database. We will test a relational database, taking the data from a pandas dataframe. We will test a non-relational database using the first database and adding documents to it.
# 
# The data base we will build is a collection of earthquake events metadata and seismograms together. Both can be two separate relational databases. We will benchmark performance on metadata manipulations.
# 
# You can find help here: http://swcarpentry.github.io/sql-novice-survey/10-prog/index.html

# In[6]:


import pandas as pd
import json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Preparing the data
# We will use the metadata of the seismic stations as a base

# In[7]:


# import the modules
import numpy as np
import pandas as pd
import io
import pickle
import requests
from datetime import datetime, timedelta
from math import cos, sin, pi, sqrt


# We will use the Northern California Earthquake Data Center stations

# In[8]:


# get the station information
url = 'http://ncedc.org/ftp/pub/doc/NC.info/NC.channel.summary.day'
s = requests.get(url).content
data = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None, skiprows=2, sep='\s+', usecols=list(range(0, 13)))
data.columns = ['station', 'network', 'channel', 'location', 'rate', 'start_time', 'end_time', 'latitude', 'longitude', 'elevation', 'depth', 'dip', 'azimuth']
data.to_csv('ncedc_stations.csv')
print(data)


# We will download earthquake waveforms from Ariane's earthquake catalog of repeating earthquakes

# ## 2. Relational database: SQLite

# This is an example on how to dump a pandas dataframe into a SQL database. But honestly, i can't seem to figure out how to query it afterwards!

# In[9]:


import sqlite3
from sqlalchemy import create_engine
engine = create_engine('sqlite:///ncedc_stations_sql.db',echo=False)
db_sql = engine.connect()
data_sql=data.to_sql('data_db_sql',db_sql,index=False,\
               if_exists='append')
data_db_sql=engine.execute("SELECT * FROM data_db_sql")

# I think that is how things work, but i can't seem to query the database...


# ## 3. Nonrelational document database: MongoDB

# In[ ]:


import pymongo
from pymongo import MongoClient

mongo_client = MongoClient('localhost', 27017)# this will create a local db (default is cloud service)

mydb=mongo_client['NCEDC']

doc = mydb['stations']
#data.reset_index(inplace=True)

data_dict = data.to_dict("records")
# Insert collection

doc.insert_many(data_dict)
print(mydb.stations.find_one())
print("   ")
print(doc)

data.to_json('ncedc_stations_mongo.json')


# Now the advantage of non-relational databases and document stores are that we can also add other files/data types into the database. We will add the earthquake catalog.

# In[11]:


namefile = 'catalog_2007_2009.pkl'
tbegin = datetime(2007, 9, 25, 0, 0, 0)
tend = datetime(2009, 5, 14, 0, 0, 0)
dt = 10.0
thresh1 = 1.4
thresh2 = 1.9
df1 = pickle.load(open(namefile, 'rb'))
df1 = df1[['year', 'month', 'day', 'hour', 'minute', 'second', 'cc', 'nchannel']]
df1 = df1.astype({'year': int, 'month': int, 'day': int, 'hour': int, 'minute': int, 'second': float, 'cc': float, 'nchannel': int})
date = pd.to_datetime(df1.drop(columns=['cc', 'nchannel']))
df1['date'] = date
df1 = df1[(df1['date'] >= tbegin) & (df1['date'] <= tend)]
df1_filter = df1.loc[df1['cc'] * df1['nchannel'] >= thresh1]
data_dict = df1_filter.to_dict("records")


# doc = mydb['stations']
doc2 = mydb['earthquakes']
doc2.insert_many(data_dict)

print(mydb.earthquakes.find_one())
print(doc)
print(doc2)


# ## 4. Benchmarking exercise

# In[12]:


import time
# from sqlalchemy import desc, select

# sorting by station nam
get_ipython().run_line_magic('time', '')
data.sort_values("station") # sort the pandas
print('Pandas sorted')

get_ipython().run_line_magic('time', '')
mydb["stations"].find().sort("station") # sort the mongoDB
print('Mongo sorted')


# In[13]:


# sorting by date of the earthquakes
get_ipython().run_line_magic('time', '')
df1_filter.sort_values("date") # sort the pandas
print('Pandas sorted')

get_ipython().run_line_magic('time', '')
mydb["earthquakes"].find().sort("date") # sort the mongoDB
print('Mongo sorted')


# In[14]:


# group by
get_ipython().run_line_magic('time', '')
data.groupby('station').station.count()
print('Pandas group by stations')

get_ipython().run_line_magic('time', '')
mydb["stations"].aggregate([\
         {"$unwind": "$station"},\
         {"$group": {"_id": "$station", "count": {"$sum": 1}}},\
  ])
print('Mongo group by station')


# In[ ]:




