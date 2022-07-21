#!/usr/bin/env python
# coding: utf-8

# # **Classification**
# 
# Problems that need a *quantitative* response (numeric value) are **regression** ; problems that need a *qualitative* response (boolean or category) are **classification**. Many statistical methods can be applied to both types of problems.
# 
# *Binary* classification have two output classes. They usually end up being "A" and "not A". Examples are "earthquake" or "no earthquake=noise". *Multiclass* classification refers to one with more than two classes. Some classifiers can handle multi class nateively (Stochastic Gradient Descent - SGD; Random Forest classification;  Naive Bayes). Others are strictly binary classifiers (Logistic Regression, Support Vector Machine classifier - SVM). 
# 
# # 1. Data download

# In[131]:


from sklearn.datasets import load_digits,fetch_openml
digits = load_digits()
digits.keys()


# The data is vector of floats. The target is an integer that is the attribute of the data. How are the data balanced between the classes? How many samples are there per class?

# In[132]:


# explore data type
data,y = digits["data"].copy(),digits["target"].copy()
print(type(data[0][:]),type(y[0]))
print(data[0][:])
print(y[0])
print(max(data[0]))
# note that we do not modify the raw data that is stored on the digits dictionary.


# In[133]:


# plot a histogram of the labels to see the balancing of the data among the classes.
plt.hist(y)


#  **how many classes are there?**
#  Since the classes are integers, we can count the number of classes using the function "unique"

# In[134]:


Nclasses = len(np.unique(y))
print(np.unique(y))
print(Nclasses)


# ### #4. Data preparation
# First print and plot the data.

# In[135]:


# plot the data
import matplotlib.pyplot as plt
# plot the first 4 data and their labels.
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


# We look at it and there is little noise and no gap. It's a nicely curated data set. I wish there were more of that for geosciences. Give us some pingos!

# ### Data re-scaling
# We could use MinMaxScaler from sklearn.preprocessing but since the formula for that is (x-min)/(max-min) and our min is 0, we could directly calculate x/max.
# (notes from https://www.kaggle.com/recepinanc/mnist-classification-sklearn)
# Note that the raw data is still stored in the dictionary ``digits`` and so we can modify the ``data`` variable in place.

# In[136]:


print(min(data[0]),max(data[0]))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(data)# fit the model for data normalization
newdata = scaler.transform(data) # transform the data. watch that data was converted to a numpy array
print(type(newdata))
print(newdata)


# ### Train-test split

# In[137]:


# Split data into 50% train and 50% test subsets
from sklearn.model_selection import train_test_split
print(f"There are {data.shape[0]} data samples")
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.5, shuffle=False)


# ## **Binary Classification**
# 
# We will first attempt to identify two classes: "5" and "not 5".

# In[138]:


y_train_5 = (y_train==5)
y_test_5 = (y_test==5)


# We will first use a classic classifier: ***Stochastic Gradient Descent SGD***.  To reproduce the results, you should set the random_state parameter.

# In[139]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
# test on the first element of the data sample and its label:
print(data[0])
print(y[0])
print("Prediction of the first data '%1.0f' onto whether it belongs to the class 5 is %s." %(y[0],sgd_clf.predict([data[0]])[0]))


# ## Classifier Performance Metrics
# 
# Confusion matrix:
# Count the instances that an element of class *A* is classified in class *B*. A 2-class confusion matrix looks like this:
# 
# | True Class      | Positive            | Negative           | Total |
# |  -------------  |  -----------------  |  --------------- | ----- |
# | Positive        | True positive: tp   | False negative: fn | p     |
# | Negative        | False positive: fp  | True negative: tn  | n     |
# | **Total**       | p'                  | n'                 | N     |
# 
# This can be extended for a multi-class classification and the matrix is KxK instead of 2x2. The best confusion matrix is one that is close to identity, with little off diagnoal terms.
# Model peformance can be assessed wih the following:
# * error = (fp+fn)/N --> 0
# * accuracy = (tp + tn)/N = 1 - error --> 1
# * tp-rate = tp/p --> 1
# * fp-rate = fp/n --> 0
# * precision =  tp/p' = tp / (tp + fp) --> 1 (but it ignores the performance in retrieving tn)
# * recall = tp/p = tp-rate = tp / (tp + fn) --> 1 (but it ignores the fact that bad values can be retrieve)
# * sensitivity = tp/p = tp-rate 
# * specificity = tn/n = 1 -fp-rate (how well do we retrieve the negatives)
# * F1 score = 2 / (1/ precision + 1/recall) = tp / (tp + (fn+fp)/2) --> 1.
# The harmonic mean of the F1 scores gives more weight to low values. F1 score is thus high if both recall and precision are high.
# 
# A good way to evaluate a model is also to use cross-validation

# In[140]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3) # predict using K-fold cross validation


# In[141]:


from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score 

print("confusion matrix")
print(confusion_matrix(y_train_5,y_train_pred))
print("precison, recall")
print(precision_score(y_train_5,y_train_pred),recall_score(y_train_5,y_train_pred))
print("F1 score")
print(f1_score(y_train_5,y_train_pred))


# In[142]:


from sklearn.metrics import classification_report
print(f"Classification report for classifier {sgd_clf}:\n"
      f"{metrics.classification_report(y_train_5, y_train_pred)}\n")


# **Precision and recall trade off**: increasing precision reduces recall. The classifier uses a *threshold* value to decide whether a data belongs to a class. Increasing the threhold gives higher precision score, decreasing the thresholds gives higher recall scores. Let's look at the various score values.

# In[143]:


y_score=sgd_clf.decision_function([data[0]])
print(y_score)


# In[144]:


from sklearn.metrics import precision_recall_curve


y_score=cross_val_predict(sgd_clf,X_train,y_train_5,cv=4,method="decision_function")
precisions,recalls,thresholds=precision_recall_curve(y_train_5,y_score)
plt.plot(thresholds,precisions[:-1])
plt.plot(thresholds,recalls[:-1],'g-')
plt.legend(['Precision','Recall'])
plt.grid(True)
plt.xlabel('Score thresholds')


# In[145]:


plt.plot(recalls[:-1],precisions[:-1])
plt.grid(True)
plt.ylabel('Precision')
plt.xlabel('Recall')


# Given the tradeoff, we can now choose a specific threshold to tune your classification. It seems that the precision drops below 90% when the recall value gets above 90% as well. So we can choose the threshold of 90%.

# In[146]:


threshold_90_precision=thresholds[np.argmax(precisions>=0.9)]
y_train_pred_90 = (y_score >=threshold_90_precision)

print(precision_score(y_train_5,y_train_pred_90))
print(recall_score(y_train_5,y_train_pred_90))


# **Receiver Operating Characteristics ROC** 
# 
# It plots the true positive rate against the false positive rate.
# The ROC curve is visual, but we can quantify the classifier performance using the *area under the curve* (aka AUC). Ideally, AUC is 1.
# 
# <div>
# <img src="roc-curve-v2-glassbox.png" width="500"/>
# </div>
# [source: https://commons.wikimedia.org/wiki/File:Roc-draft-xkcd-style.svg]

# In[147]:


from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_train_5,y_score)
plt.plot(fpr,tpr,linewidth=2);plt.grid(True)
plt.plot([0,1],[0,1],'k--')


# Compare with another classifier method. We will try ***Random Forest*** and compare the two classifiers. Instead of outputing scores, RF works with probabilities. So the value returned as between 0 and 1 with the probability of appartenance to the given class.

# In[150]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42) # model design
y_rf_5 = cross_val_predict(rf_clf,X_train,y_train_5,cv=3,method="predict_proba")
y_scores_rf = y_rf_5[:,1] # score in the positive class
fpr_rf,tpr_rf,threshold_rf = roc_curve(y_train_5,y_scores_rf)


# In[151]:


plt.plot(fpr_rf,tpr_rf,'r',linewidth=2)
plt.plot(fpr,tpr,linewidth=2);plt.grid(True)
plt.plot([0,1],[0,1],'k--')


# ## Multiclass Classification
# 
# Here we will use several well known classifiers: Support Vector Machine, k-nearest neighbors, Stochastic Gradient Descent

# In[25]:


import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Support Vector Machine classifier
clf = SVC(gamma=0.001) # model design
clf.fit(X_train, y_train) # learn
svc_prediction = clf.predict(X_test) # predict on test
print("SVC Accuracy:", metrics.accuracy_score(y_true=y_test ,y_pred=svc_prediction))

# K-nearest Neighbors
knn_clf = KNeighborsClassifier() # model design
knn_clf.fit(X_train, y_train) # learn
knn_prediction = knn_clf.predict(X_test) # predict on test
print("K-nearest Neighbors Accuracy:", metrics.accuracy_score(y_true=y_test ,y_pred=knn_prediction))

# Random Forest
rf_clf = RandomForestClassifier(random_state=42, verbose=True) # model design
rf_clf.fit(X_train, y_train)# learn
rf_prediction = rf_clf.predict(X_test) # predict on test
print("Random Forest Accuracy:", metrics.accuracy_score(y_true=y_test ,y_pred=rf_prediction))


# In[26]:


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, rf_prediction):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')


# In[27]:


print("Support Vector Machine")
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, svc_prediction)}\n")
disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()


# In[28]:


print("K-nearest neighbors")
print(f"Classification report for classifier {knn_clf}:\n"
      f"{metrics.classification_report(y_test, knn_prediction)}\n")
disp = metrics.plot_confusion_matrix(knn_clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()


# In[29]:


print("Random Forest")
print(f"Classification report for classifier {rf_clf}:\n"
      f"{metrics.classification_report(y_test, rf_prediction)}\n")
disp = metrics.plot_confusion_matrix(rf_clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()


# In[30]:


from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay


# ### Multiclass classification

# In[44]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import svm

from sklearn.metrics import roc_curve, auc

random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))

y = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9])

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.5, shuffle=False)

y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(Nclasses):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Plot of a ROC curve for a specific class
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
for i in range(Nclasses):
    plt.plot(fpr[i], tpr[i], label='ROC label %1.0f (area = %0.2f)' % (i,roc_auc[i]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")


# In[ ]:




