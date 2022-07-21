#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction
# 
# Ideally, one would not need to extract or select feature in the input data. However, reducing the dimensionality as a separate pre-processing steps may be advantageous:
# 
# 1. The complexity of the algorithm depends on the number of input dimensions and size of the data.
# 2. If some features are unecessary, not extracting them saves compute time
# 3. Simpler models are more robust on small datasets
# 4. Fewer features lead to a better understanding of the data.
# 5. Visualization is easier in fewer dimensions.
# 
# 
# Feature *selection* finds the dimensions that explain the data without loss of information and ends with a smaller dimensionality of the input data. A *forward selection* approach starts with one variable that decreases the error the most and add one by one. A *backward selection* starts with all variables and removes them one by one.
# 
# Feature *extraction* finds a new set of dimension as a combination of the original dimensions. They can be supervised or unsupervised depending on the output information. Examples are **Principal Component Analysis**, **Linear Discriminant Analysis**
# 
# 
# ## Principal Component Analysis
# 
# PCA is an unsupervised learning method that finds the mapping from the input to a lower dimemsional space with minimum loss of information. First, we will download again our GPS time series from Oregon.

# In[21]:


# Import useful modules
import requests, zipfile, io, gzip, glob, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


# Download data and save into a pandas.
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


# In[23]:


# Extract the three data dimensions
E = np.asarray(ue)# East component
N = np.asarray(un)# North component
U = np.asarray(uv)# Vertical component


fig=plt.figure(figsize=(11,8))
ax=fig.add_subplot(projection='3d')
# viridis = cm.get_cmap('viridis', len(E))
# print(viridis)
# ax.scatter(E,N,U,cmap=viridis,norm=True);ax.grid(True)
ax.scatter(E,N,U);ax.grid(True)
ax.set_xlabel('East motions (mm)')
ax.set_ylabel('North motions (mm)')
ax.set_zlabel('Vertical motions (mm)')


# PCA idenditfies the axis that accounts for the largest amount of variance in the training set. PCA uses the *(Singular Value Decomposition (SVD)* to decompose the training set matrix $\mathbf{X} $:
# 
# $\mathbf{X} = \mathbf{U} \Sigma \mathbf{V}^T ,$
# 
# where $\mathbf{V}^T$ contains the eigenvectors, or principal components. The *PCs* are normalized, centered around zero
# 
# The *1st principal component* eigenvector that has the highest eigenvalue in the direction that has the highest variance.

# In[58]:


# Organize your data in a matrix
X = np.vstack([E,N,U]).transpose()
print(U.shape)
print(X.shape)


# In *PCA*, the input data is centered but not scaled for each feature before applying the SVD.
# 
# 

# In[88]:


from sklearn.decomposition import PCA
pca=PCA(n_components=3).fit(X)# retain all 3 components
print(pca)


# In[89]:


# The 3 PCs
print(pca.components_)


# In[90]:


# The 3 PCs' explained variance
print(pca.explained_variance_)


# In[91]:


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
v= pca.components_[0,:] * 3 * pca.explained_variance_[0]
print(pca.mean_[:-1])
print(v[0])
draw_vector(pca.mean_[:-1], pca.mean_[:-1] + v[0])
plt.axis('equal');


# In[97]:


# The 3 PCs' explained variance ratio: how much of the variance is explained by each component
print(pca.explained_variance_ratio_)
fig,ax=plt.subplots(2,1,figsize=(11,8))
ax[0].plot(pca.explained_variance_ratio_)
ax[0].set_xlabel('Number of dimensions')
ax[0].set_ylabel('Explained variance ')
ax[0].set_title('Variance explained with each PC')
ax[1].plot(np.cumsum(pca.explained_variance_ratio_))
ax[1].set_xlabel('Number of dimensions')
ax[1].set_ylabel('Explained variance ')
ax[1].set_title('Variance explained with cumulated PCs')


# Now we see that 97% of the variance is explained by the first PC. We can reduce the dimension of our system by only working with a single dimension. Another way to choose the number of dimensions is to select the minimum of PCs that would explain 95% of the variance.

# In[100]:


d = np.argmax(np.cumsum(pca.explained_variance_ratio_)>=0.95) +1
print("minimum dimension size to explain 95% of the variance ",d)


# In[101]:


pca = PCA(n_components=d).fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)


# In[ ]:


# Find the azimuth of the displacement vector
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');

# Print out the azimuth and norm 
import math
azimuth=math.degrees(math.atan2(pca.components_[0][0],pca.components_[0][1]))
if azimuth <0:azimuth+=360
print("direction of the plate ",azimuth," degrees from North")


# *SVD* can be computationally intensive for larger dimensions. It is recommended to use a **randomized PCA** to approximate the first principal components. Scikit-learn automatically switches to randomized PCA in either the following happens: data size > 500, number of vectors is > 500 and the number of PCs selected is less than 80% of either one.

# ## Independent Component Analysis
# 
# ICA is used to estimate sources given noisy measurements. It is frequently used in Geodesy to isolate contributions from earthquakes and hydrology. 

# In[122]:


from scipy import signal
from sklearn.decomposition import FastICA

# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# create 3 source signals
s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal


S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise
S /= S.std(axis=0)  # Standardize data

print(S)



# In[124]:


# Mix data
# create 3 signals at 3 receivers:
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations


# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
print(A_,A)
# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components


# In[126]:


plt.figure(figsize=(11,8))
models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals', 
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)
plt.tight_layout()
plt.show()


# ## Other Techniques
# 
# 
# 1. Random Projections
# https://scikit-learn.org/stable/modules/random_projection.html
# 2. Multidimensional Scaling
# 3. Isomap
# 4. t-Distributed stochastic neighbor embedding
# 5. Linear discriminant analysis
