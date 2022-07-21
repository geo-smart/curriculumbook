#!/usr/bin/env python
# coding: utf-8

# # K-means clustering - Homework
# 
# In the tutorial, we have seen how to choose the number of clusters using the elbow method. However, we have also noticed that it does not always work very well. Let us study another method based on the prediction strength.
# 
# To know more about it, you can read the paper: Tibshirani, R. and Walther, G. (2005) Cluster validation by prediction strength. Journal of Computational and Graphical Statistics 14(3):511-528.
# 
# In this homework, we are going to  use the function KMeans from the SciKitLearn package. See here for the documentation:
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

# ## Prediction strength
# 
# Suppose we have a dataset of $X = \left\{ x_{i , j} \right\}$ of $n$ observations of $d$-dimensional variables. Let us divide these observations into a train set $X_{tr}$ and a test set $X_{te}$ of size $n_{tr}$ and $n_{te}$ respectively.
# 
# Let us choose the number of clusters $k$, and apply clustering to both the training data and the test data independently.
# 
# Let us now denote $A_{1} , A_{2} , \cdots , A_{k}$ the indices of the test observations in the test clusters $1 , 2 , \cdots , k$, and $n_{1} , n_{2} , \cdots , n_{k}$ the number of observations in these clusters.
# 
# We now consider the clusters obtained with the training data, and denote this classifying operation $C \left( X_{tr} \right)$. We now apply this classifying operation to the test set. 
# 
# Let us now denote $D_j \left[ C \left( X_{tr} , k \right) , X_{te} \right]$ the $n_{te}$ by $n_{te}$ matrix which $i i'$ element $D_j \left[ C \left( X_{tr} , k \right) , X_{te} \right] _{i i'}$ is equal to $1$ if observations $i$ and $i'$ from the $j$th cluster of the test set fall into the same training set cluster, and $0$ otherwise. The prediction strength is then defined by:
# 
# $ps \left( k \right) = \min_{ 1 \leq j \leq k} \frac{1}{n_{j} \left( n_{j } - 1 \right)} \sum_{i \neq i' \in A_{j}} D_j \left[ C \left( X_{tr} , k \right) , X_{te} \right] _{i i'}$ (**eq 1**)

# ## Data gathering and cleaning

# Import useful Python packages

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from math import cos, sin, pi, sqrt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Set seed for reproducibility

# In[ ]:


random.seed(0)


# Import data from the PNSN earthquake catalog.

# In[ ]:


catalog = pd.read_csv('pnsn_catalog.csv')
catalog.drop(columns=['Evid', 'Magnitude', 'Magnitude Type', 'Epoch(UTC)', 'Time UTC', 'Time Local', 'Distance From', 'Depth Mi'], inplace=True)
catalog.columns = ['latitude', 'longitude', 'depth']


# Apply PCA and normalization.

# In[ ]:


data = catalog.to_numpy()
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
scaler = preprocessing.StandardScaler().fit(data_pca)
data_scaled = scaler.transform(data_pca)


# ## Homework

# ### Question 1 (1 point)
# 
# Write code to divide the data into a training set and a test set of approximately the same size.

# In[ ]:





# ### Question 2 (2 points)
# 
# For now, we choose to have k = 2 clusters.
# 
# Write code to apply K-means clustering to the training set and the test set using the Kmeans function from ScikitLearn.

# In[ ]:





# ### Question 3 (2 points)
# 
# Get the clusters for the test set.

# In[ ]:





# Plot the data from the test set with two different colors for the two clusters.

# In[ ]:





# ### Question 4 (2 points)
# 
# Use the clustering and centroids from the training set to predict to which cluster the data points from the test set should belong.

# In[ ]:





# Plot the data from the test set with two different colors for the two clusters.

# In[ ]:





# ### Question 5
# 
# Compute the prediction strength for $k$ = 2 as defined at the beginning. Hint: use **eq 1** with nested loops

# In[ ]:





# ### Question 6
# 
# Write a function that does steps 1 to 5 for any number $k$ of clusters and return the prediction strength or a given $k$ number of clusters. 

# In[ ]:





# ### Question 7
# 
# Apply this function to $k = 2, \cdots , 20$.

# In[ ]:





# ### Question 8
# 
# Plot the prediction strength as a function of number of clusters. What is the optimal number of clusters for this dataset?

# In[ ]:




