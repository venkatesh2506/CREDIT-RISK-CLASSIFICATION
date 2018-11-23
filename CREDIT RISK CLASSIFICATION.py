
# coding: utf-8

# # Credit Risk Classification

# Here we have a dataset consists of 1000 Rows representing the persons who takes a credit from the bank and Each person is classified as good or bad according to the given attributes

# Now we are going to build a model by using Machine learning algorithms.This gonna be achived by performing following steps.....

# ## 1)Importing Libraries

# In order to analyze and build a model on dataset we require some python libraries.First we need to import them

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## 2)Reading the Dataset

# In[2]:


df_credit = pd.read_csv("german_credit_data.csv")


# ## 3)Understanding the Data

# ### (1)Summerization

# In[3]:


df_credit.head()


# In[4]:


df_credit.info()


# In[5]:


df_credit.describe()


# In[6]:


df_credit.kurt()


# In[7]:


df_credit.skew()


# From the above Summerization results we can say that our data quality is good becauese all the summarry values looks good and also both the skewness and kurtosis are in their range of -0.8 to +0.8 and -3 to +3

# ### (2)Visualization

# Visuallization is the gretest method to easily understand the huge amount of data and we can easily grab the insights from it.

# In[8]:


sns.countplot('Sex',data = df_credit)


# In[9]:


plt.subplot(221)
sns.countplot('Housing',data = df_credit)
plt.subplot(222)
sns.countplot('Saving accounts',data = df_credit)
plt.subplot(223)
sns.countplot('Checking account',data = df_credit)
plt.subplot(224)
sns.countplot('Sex',data = df_credit)


# ## 4)Preprocessing the data

# In this dataset we are having several features which are of Categorical type,it is difficult for a scikit-learn library to understand the categorical varibles.To overcome this we have to convert them into as numerical ones.This can be achived in this preprocessing step.

# In[10]:


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
df_credit["Risk"] = lb.fit_transform(df_credit["Risk"])
df_credit["Sex"] = lb.fit_transform(df_credit["Sex"])


# Label Binarizer can be used to convert the categorical data into numerical data of values 0 and 1

# In[11]:


interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Youth', 'Adult', 'Senior']
df_credit["Age"] = pd.cut(df_credit.Age, interval, labels=cats)


# Age is not a regular kind of numerical value it is better to be in the form of intervals Of numerical kind.Which can be easily done by using cut function from the Pandas library.

# In[12]:


df_credit.head()


# Here our feature varibles[saving accounts,cheacking accounts] having some null values.It is not possible to code them when null values are there.so we are going to replace them with the no_inf 

# In[13]:


df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')


# Our remaining categorical featurs contains more than 2 possible outcomes those can bee easily decoded by using Label Encoder from the scikit learn library.

# In[14]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_credit["Housing"]= lb.fit_transform(df_credit["Housing"])
df_credit["Age"]=lb.fit_transform(df_credit["Age"])
df_credit["Saving accounts"]= lb.fit_transform(df_credit["Saving accounts"])
df_credit["Checking account"]= lb.fit_transform(df_credit["Checking account"])
df_credit["Duration"]= lb.fit_transform(df_credit["Duration"])
df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
del df_credit["Purpose"]
del df_credit["Unnamed: 0"]


# In[15]:


df_credit.head()


# Preparing our Feature and target datasets

# In[16]:


X= df_credit.drop("Risk", axis= 1)
y= df_credit["Risk"]


# In order to achive our classification model accurately we have to maintain our data in similar scale.For this purpose we have a pre-processing technique called Standard scaler in the scikit-learn library

# In[17]:


from sklearn.preprocessing import StandardScaler
SC= StandardScaler()
X= SC.fit_transform(X)
X=pd.DataFrame(X)
X


# # 5) Building Models

# Now our data is in perfect form to build a model so we are going to import our requied libraries from the Scikit learn.

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Spliting our data into training data set and testing data set of 75-25 combo.

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)


# we are going to build a Logistic Regression model on the dataset.Logistics Regression is the better option for the classification problems with two possible outcomes.Here also our target varible consists of two posible outcomes os Good and Bad.

# In[20]:


logreg = LogisticRegression()
logreg.fit(X_train ,y_train)
y_pred = logreg.predict(X_test)


# Building ROC curve

# In[21]:


from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--1')
plt.plot(fpr,tpr,label = 'Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Curve')


# Calculating the ROC score to find the performance of the model.If area under ROC curve is high,then model is good otherwise termed it as bad

# In[22]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_prob)


# Implementing Decision Tree model and finding its scores

# In[23]:


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print(tree.score(X_train,y_train))
print(tree.score(X_test,y_test))


# Implementing Random forest model and finding its scores

# In[24]:


forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)


# In[25]:


print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))


# Implementing Gradient boost classifier model and finding its scores

# In[26]:


gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))


# From the above models and their performance scores we can go with Logstic Regression Model
