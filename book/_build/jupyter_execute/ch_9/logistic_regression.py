#!/usr/bin/env python
# coding: utf-8

# # Logistic regression
# 
# _Warning!_ Although it is called logistic _regression_, logistic regression is actually a _classification_ method.

# In this lab, we are going to talk about:
# - A simple classification method: Logistic regression
# - Gradient descent method
# - Automatic differentiation
# - Introduction to PyTorch

# ## Logistic regression

# Remember linear regression:
# 
# $y = b + x w + \epsilon$
# 
# $y$ is a vector of length $n$, $x$ is a matrix with $n$ rows and $p$ columns, corresponding to $n$ observations and $p$ features that are used to explain $y$. 
# 
# $b$ is a scalar. $w$ is a vector of length $p$. $\epsilon$ is a random error vector, of length $n$. It is independent of $x$, and has mean zero.
# 
# Our objective is to find the best values of $b$ and $w$ so that the values of $\hat{y} = b + x w$ are as close as possible to the actual values $y$.
# 
# For linear regression, $y$ is a quantitative variable. What if $y$ is a qualitative variable, for example $y = 0$ for "no", and $y = 1$ for "yes"?

# One way to use regression to solve a classification problem is to model the probability of the variable $y$ taking the value 1:
# 
# $P (y = 1) = b + x w$
# 
# Once we have found the best values $\hat{b}$ and $\hat{w}$, we compute $\hat{y} = \hat{b} + x \hat{w}$. If $\hat{y} \geq 0.5$, we decide to classify this observation as $y = 1$, that is "yes". If $\hat{y} < 0.5$, we decide to classify this observation as $y = 0$, that is "no".

# There is a problem with this method. We would like to have $0 \leq P (y = 1) \leq 1$ because it is a probability. However, there is nothing in this formulation that forces $b$ and $w$ to take values such that $\hat{y} = \hat{b} + x \hat{w}$ will always takes values in $[0, 1]$.

# To solve this problem, we can instead write:
# 
# $z = b + x w$ and $P (y = 1) = \frac{1}{1 + e^{-z}}$
# 
# That way, we always have $0 \leq P (y = 1) \leq 1$. When $b + x w$ gets large, $P (y = 1)$ gets close to 1, and the value "yes" is more and more likely. When $b + x w$ gets small, $P (y = 1)$ gets close to 0, and the value "no" is more and more likely.

# How do we find the optimal value of $b$ and $w$? We define the cross-entropy loss function. For one observation, the loss function is:
# 
# $\mathcal{L} = - (y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i)$ with $\hat{y}_i = \frac{1}{1 + e^{- (b + x_i^T w)}}$ where $x_i$ is the $i$th row of x.
# 
# If the true observation $y_i$ is 1 ("yes") and $\hat{y}_i = 1$, the loss function takes the value 0. If $\hat{y}_i = 0$, the loss function tends to infinity.
# 
# If the true observation $y$ is 0 ("no") and $\hat{y}_i = 0$, the loss function takes the value 0. If $\hat{y}_i = 1$, the loss function tends to infinity.
# 
# For all the $n$ observations, we write:
# 
# $\mathcal{L} = \sum_{i = 1}^n \mathcal{L}_i$
# 
# Our objective is thus to find the values of $b$ and $\omega$ that minimize the loss function. Note that with this formulation, $\mathcal{L}$ is always positive.

# ## Gradient descent
# 
# We know that the gradient $\frac{\partial \mathcal{L}}{\partial w_j}$ is positive if the loss $\mathcal{L}$ increases when $w_j$ increases. Reversely, the gradient $\frac{\partial \mathcal{L}}{\partial w_j}$ is negative if the loss $\mathcal{L}$ decreases when $w_j$ increases.

# To obtain smaller and smaller values of the loss, at each iteration we take:
# 
# $w_j^{(k + 1)} = w_j^{(k)} - \alpha \frac{\partial \mathcal{L}}{\partial w_j}$ for $j = 1 , \cdots , p$
# 
# $b^{(k + 1)} = b^{(k)} - \alpha \frac{\partial \mathcal{L}}{\partial b}$
# 
# We assume that the value of $\alpha$ is not too big. If the gradient is positive, then the value of $w_j$ will decrease at each iteration, and the value of the loss function will decrease. If the gradient is negative, then the value of $w_j$ will increase at each iteration, and the value of the loss will decrease.

# So now, all we need to do is to compute the gradient of the loss function. 

# ## Automatic differentiation

# There are three ways of computing the gradient. The first method is to use the formula of the loss:
# 
# $\mathcal{L} (w_j , b) = - \sum_{i = 1}^n y_i \log (\frac{1}{1 + \exp (- b - \sum_{j = 1}^p w_j x_{i,j})}) + (1 - y_i) \log (1 - \frac{1}{1 + \exp (- b - \sum_{j = 1}^p w_j x_{i,j})})$
# 
# and to calculate the exact formula of the derivatives $\frac{\partial \mathcal{L}}{\partial w_j}$ and $\frac{\partial \mathcal{L}}{\partial b}$. You just then have to implement the exact formula in the code to compute the gradient.
# 
# When the formula gets more and more complicated, you become more and more likely to make a mistake, either in the calculation of the derivative formula, either in the implementation in your code.

# The second method is to compute an approximation of the gradient:
# 
# $\frac{\partial \mathcal{L}}{\partial w_j} = \frac{\mathcal{L}(w_j + \Delta w_j) - \mathcal{L}(w_j)}{\Delta W_j}$
# 
# If you write too many approximations, the method may not work very well and give inexact results.

# The third method is to use automatic differentiation. If we write:
# 
# $z = x_i^T w + b = f_x(w, b)$, $\sigma = \frac{1}{1 + e^{-z}} = g(z)$ and $L = - (y_i \log(\sigma) + (1 - y_i) \log(1 - \sigma)) = h_y(\sigma)$, we get:
# 
# $\frac{\partial L}{\partial w_j} = \frac{\partial f}{\partial w_j} g'(z) h'(\sigma)$

# It is very easy to compute the exact formula of the derivatives:
# 
# $\frac{\partial f}{\partial w_j}(w, b) = x_{i,j}$
# 
# $g'(z) = \frac{e^{-z}}{(1 + e^{-z})^2}$
# 
# $h'(\sigma) = - \frac{y_i}{\sigma} + \frac{1 - y_i}{1 - \sigma}$
# 
# When computing $L$, we thus need to keep in memory the values of $\frac{\partial f}{\partial w_j}(w, b)$, $g'(z)$, and $h'(\sigma)$ to be able to compute the gradient. That is what PyTorch is doing.

# ## Introduction to PyTorch

# PyTorch (https://pytorch.org/) is a Python package which allows you to build and train neural networks. It is based on automatic differentation.

# In[1]:


import torch
import numpy as np
import pandas as pd
from math import exp
from sklearn import preprocessing


# Let us import a dataset as an example. This example has been downloaded from Kaggle: https://www.kaggle.com/adityakadiwal/water-potability

# In[2]:


data = pd.read_csv('water_potability.csv')
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
x = data.drop(columns=['Potability']).to_numpy()
y = data.Potability.to_numpy()
N = len(data)


# First we need to nomalize the data

# In[3]:


scaler = preprocessing.StandardScaler().fit(x)
x = scaler.transform(x)


# We are going to compute the loss corresponding to the first observation in the dataset. Instead of using Numpy arrays to put our data and parameters, we are going to use torch tensors, because they have properties that Numpy arrays do not have.
# 
# This is the features of the first observation:

# In[4]:


x_i = torch.torch.from_numpy(x[0, :])
x_i = x_i.float()


# This is the class of the first observations:

# In[5]:


y_i = y[0]


# Let us take random values for $w$ and $b$. When creating these variables, we use the option requires_grad=True because we will later want to compute the gradient with respect to these variables.

# In[6]:


W = torch.rand(9, requires_grad=True)
B = torch.rand(1, requires_grad=True)


# This function will be used to sepcify that we will want to compute the gradient with respect to the variable var.

# In[7]:


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


# Let us define $z = f(w, b) = x_i^T w + b$. We have $\frac{\partial f}{\partial w_j} = x_{i,j}$ and $\frac{\partial f}{\partial b} = 1$.

# In[8]:


z = W.dot(x_i) + B


# In[9]:


z.register_hook(set_grad(z))


# Let us define $\sigma = g(z) = \frac{1}{1 + e^{-z}} = g(f(w, b)) = (g \circ f) (w, b)$. We have $g'(z) = \frac{e^{-z}}{(1 + e^{-z})^2}$.

# In[10]:


sigma = 1.0 / (1.0 + torch.exp(- z))


# Note that here we use the function torch.exp instead of numpy.exp. That is because numpy just calculate the value of $e^x$ but does not know that the derivative of $e^x$ is $e^x$. If we want to be able to use automatic differentiation, we need to use the equivalent torch function that will compute both $\sigma(z)$ and $\frac{\partial \sigma}{\partial z}(z)$. This last value will be necessary and we will later compute the gradient.

# In[11]:


sigma.register_hook(set_grad(sigma))    


# Let us define $L = h(\sigma) = - (y_i \log(\sigma) + (1 - y_i) \log(1 - \sigma)) = h(g(z)) = h(g(f(w, b))) = (h \circ g \circ f) (w, b)$. We have $L'(\sigma) = - (\frac{y_i}{\sigma} - \frac{1 - y_i}{1 - \sigma})$.

# In[12]:


L = - (y_i * torch.log(sigma) + (1 - y_i) * torch.log(1 - sigma))


# We can now compute the gradient of the loss for one observation. This command compute the gradient of L with respect to all the variables for which I asked to compute the gradient, that is W, B, z, and sigma, but it does not return the value.

# In[13]:


L.backward()


# We have $\frac{\partial L}{\partial \sigma} = - (\frac{y_i}{\sigma} - \frac{1 - y_i}{1 - \sigma})$. Let us compute the result with PyTorch and using the exact mathematical formula.

# In[14]:


print(sigma.grad, - (y_i / sigma - (1 - y_i) / (1 - sigma)))


# We have $\frac{\partial L}{\partial z} = g'(z) h'(g(z))$ that is $\frac{\partial L}{\partial z} = g'(z) h'(\sigma)$. Let us compute the result with PyTorch and using the exact mathematical formula.

# In[15]:


print(z.grad, (exp(-z) / ((1 + exp(-z)) ** 2.0)) * (- (y_i / sigma - (1 - y_i) / (1 - sigma))))


# We have $\frac{\partial L}{\partial b} = \frac{\partial f}{\partial b} g'(f(w, b)) h'(g(f(w, b)))$ that is $\frac{\partial L}{\partial b} = \frac{\partial f}{\partial b} g'(z) h'(\sigma)$. Similarly, we have $\frac{\partial L}{\partial w_j} = \frac{\partial f}{\partial w_j} g'(f(w, b)) h'(g(f(w, b)))$ that is $\frac{\partial L}{\partial w_j} = \frac{\partial f}{\partial w_j} g'(z) h'(\sigma)$. Let us compute the result with PyTorch and using the exact mathematical formula.

# In[16]:


print(B.grad, 1 * (exp(-z) / ((1 + exp(-z)) ** 2.0)) * (- (y_i / sigma - (1 - y_i) / (1 - sigma))))


# In[17]:


print(W.grad, x_i * (exp(-z) / ((1 + exp(-z)) ** 2.0)) * (- (y_i / sigma - (1 - y_i) / (1 - sigma))))


# ## Implementation of logistic regression

# Let us now implement logistic regression using the whole dataset.

# In[18]:


X = torch.torch.from_numpy(x)
X = X.float()


# In[19]:


Y = torch.torch.from_numpy(y)
Y = Y.float()


# This is the code for one iteration of the gradient descent algorithm.

# In[20]:


def step(X, Y, W, B, alpha):
    # Compute the loss for the current value of W and B
    z = X @ W + B
    sigma = 1.0 / (1.0 + torch.exp(- z))
    L = torch.sum(- (Y * torch.log(sigma) + (1 - Y) * torch.log(1 - sigma)))
    # Compute the gradient of the loss
    L.backward()
    # Specifically, we want the gradient with respect to W and B
    dW = W.grad
    dB = B.grad
    # Update the values of W and B
    W = W - alpha * dW
    W.retain_grad()
    B = B - alpha * dB
    B.retain_grad()
    # Return the new values of W and B
    return (W, B, L)


# We can now implement the logistic regression:

# In[21]:


def logistic_regression(X, Y, alpha, max_iter, epsilon):
    p = X.size()[1]
    # We initiate W and B with random values
    W = torch.rand(p, requires_grad=True)
    B = torch.rand(1, requires_grad=True)
    i_iter = 0
    dL = 2 * epsilon
    # We iterate until we reach the maximum number of iterations or the loss no longer decreases
    while ((i_iter < max_iter) and (dL > epsilon)):
        if i_iter > 0:
            L_old = L
        (W, B, L) = step(X, Y, W, B, alpha)
        if i_iter > 0:
            dL = abs((L - L_old) / L_old)
        i_iter = i_iter + 1
    return (W, B, L)


# Let us now run our code:

# In[22]:


(W, B, L) = logistic_regression(X, Y, 0.001, 200, 0.001)


# Let us now try to make predictions on the training test:

# In[23]:


yhat = 1.0 / (1.0 + torch.exp(- (X @ W + B)))


# We convert the torch tensor to a Numpy array:

# In[24]:


yhat = yhat.detach().numpy()


# We transform the values of the probability (between 0 and 1) into the value of the class to which each observation belongs (here 0 or 1):

# In[25]:


yhat = np.where(yhat > 0.5, 1, 0)


# Let us now compute some classification metrics.

# In[26]:


# True positive
tp = np.sum((y == 1) & (yhat == 1))
print(tp)


# In[27]:


# False negative
fn = np.sum((y == 1) & (yhat == 0))
print(fn)


# In[28]:


# False positive
fp = np.sum((y == 0) & (yhat == 1))
print(fp)


# In[29]:


# True negative
tn = np.sum((y == 0) & (yhat == 0))
print(tn)


# In[30]:


# Accurracy (percentage of correct classifications)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(accuracy)


# In[31]:


# Recall (= Sensitivity = percentage of positive value correctly classified)
recall = tp / (tp + fn)
print(recall)


# In[32]:


# Precision (= percentage of positive predictions that were correct)
precision = tp / (tp + fp)
print(precision)


# In[33]:


# F1
F1 = (2 * precision * recall) / (precision + recall)
print(F1)


# ## Appendix
# 
# Logistic regression is a nice example to start learning about automatic differentiation and PyTorch. However, if you actually want to use logistic regression for your own dataset, it is much easier to use the function already existing in ScikitLearn:

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


# In[35]:


model = LogisticRegression(random_state=0).fit(x, y)


# In[36]:


model.coef_


# In[37]:


model.intercept_


# In[38]:


yhat = model.predict(x)


# In[39]:


metrics = precision_recall_fscore_support(y, yhat, average='binary')


# In[40]:


(precision, recall, F1) = (metrics[0], metrics[1], metrics[2])
print(precision, recall, F1)


# In[ ]:




