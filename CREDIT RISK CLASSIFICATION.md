
# Credit Risk Classification

Here we have a dataset consists of 1000 Rows representing the persons who takes a credit from the bank and Each person is classified as good or bad according to the given attributes

Now we are going to build a model by using Machine learning algorithms.This gonna be achived by performing following steps.....

## 1)Importing Libraries

In order to analyze and build a model on dataset we require some python libraries.First we need to import them


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## 2)Reading the Dataset


```python
df_credit = pd.read_csv("german_credit_data.csv")
```

## 3)Understanding the Data

### (1)Summerization


```python
df_credit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>bad</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
      <td>good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
      <td>bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_credit.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 11 columns):
    Unnamed: 0          1000 non-null int64
    Age                 1000 non-null int64
    Sex                 1000 non-null object
    Job                 1000 non-null int64
    Housing             1000 non-null object
    Saving accounts     817 non-null object
    Checking account    606 non-null object
    Credit amount       1000 non-null int64
    Duration            1000 non-null int64
    Purpose             1000 non-null object
    Risk                1000 non-null object
    dtypes: int64(5), object(6)
    memory usage: 86.0+ KB
    


```python
df_credit.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Age</th>
      <th>Job</th>
      <th>Credit amount</th>
      <th>Duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>499.500000</td>
      <td>35.546000</td>
      <td>1.904000</td>
      <td>3271.258000</td>
      <td>20.903000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>288.819436</td>
      <td>11.375469</td>
      <td>0.653614</td>
      <td>2822.736876</td>
      <td>12.058814</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>250.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>249.750000</td>
      <td>27.000000</td>
      <td>2.000000</td>
      <td>1365.500000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>499.500000</td>
      <td>33.000000</td>
      <td>2.000000</td>
      <td>2319.500000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>749.250000</td>
      <td>42.000000</td>
      <td>2.000000</td>
      <td>3972.250000</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999.000000</td>
      <td>75.000000</td>
      <td>3.000000</td>
      <td>18424.000000</td>
      <td>72.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_credit.kurt()
```




    Unnamed: 0      -1.200000
    Age              0.595780
    Job              0.501891
    Credit amount    4.292590
    Duration         0.919781
    dtype: float64




```python
df_credit.skew()
```




    Unnamed: 0       0.000000
    Age              1.020739
    Job             -0.374295
    Credit amount    1.949628
    Duration         1.094184
    dtype: float64



From the above Summerization results we can say that our data quality is good becauese all the summarry values looks good and also both the skewness and kurtosis are in their range of -0.8 to +0.8 and -3 to +3

### (2)Visualization

Visuallization is the gretest method to easily understand the huge amount of data and we can easily grab the insights from it.


```python
sns.countplot('Sex',data = df_credit)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15d78c624e0>




![png](output_18_1.png)



```python
plt.subplot(221)
sns.countplot('Housing',data = df_credit)
plt.subplot(222)
sns.countplot('Saving accounts',data = df_credit)
plt.subplot(223)
sns.countplot('Checking account',data = df_credit)
plt.subplot(224)
sns.countplot('Sex',data = df_credit)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15d7903b4a8>




![png](output_19_1.png)


## 4)Preprocessing the data

In this dataset we are having several features which are of Categorical type,it is difficult for a scikit-learn library to understand the categorical varibles.To overcome this we have to convert them into as numerical ones.This can be achived in this preprocessing step.


```python
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
df_credit["Risk"] = lb.fit_transform(df_credit["Risk"])
df_credit["Sex"] = lb.fit_transform(df_credit["Sex"])
```

Label Binarizer can be used to convert the categorical data into numerical data of values 0 and 1


```python
interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Youth', 'Adult', 'Senior']
df_credit["Age"] = pd.cut(df_credit.Age, interval, labels=cats)
```

Age is not a regular kind of numerical value it is better to be in the form of intervals Of numerical kind.Which can be easily done by using cut function from the Pandas library.


```python
df_credit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Senior</td>
      <td>1</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Student</td>
      <td>0</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Adult</td>
      <td>1</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Adult</td>
      <td>1</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Adult</td>
      <td>1</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Here our feature varibles[saving accounts,cheacking accounts] having some null values.It is not possible to code them when null values are there.so we are going to replace them with the no_inf 


```python
df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')
```

Our remaining categorical featurs contains more than 2 possible outcomes those can bee easily decoded by using Label Encoder from the scikit learn library.


```python
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
```


```python
df_credit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Risk</th>
      <th>Purpose_car</th>
      <th>Purpose_domestic appliances</th>
      <th>Purpose_education</th>
      <th>Purpose_furniture/equipment</th>
      <th>Purpose_radio/TV</th>
      <th>Purpose_repairs</th>
      <th>Purpose_vacation/others</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1169</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5951</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2096</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7882</td>
      <td>26</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4870</td>
      <td>17</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Preparing our Feature and target datasets


```python
X= df_credit.drop("Risk", axis= 1)
y= df_credit["Risk"]
```

In order to achive our classification model accurately we have to maintain our data in similar scale.For this purpose we have a pre-processing technique called Standard scaler in the scikit-learn library


```python
from sklearn.preprocessing import StandardScaler
SC= StandardScaler()
X= SC.fit_transform(X)
X=pd.DataFrame(X)
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.465817</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.955847</td>
      <td>-1.344000</td>
      <td>-0.745131</td>
      <td>-1.558464</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.286714</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>0.949817</td>
      <td>2.032467</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>-0.416562</td>
      <td>-0.760479</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>3.993639</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-2.016956</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>1.634247</td>
      <td>1.633475</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-2.016956</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>0.566664</td>
      <td>0.436498</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-2.016956</td>
      <td>0.955847</td>
      <td>0.813303</td>
      <td>2.050009</td>
      <td>1.234482</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>3.993639</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>1.787019</td>
      <td>0.813303</td>
      <td>-0.154629</td>
      <td>0.436498</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>1.677670</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>1.303197</td>
      <td>1.234482</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.465817</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>2.618191</td>
      <td>0.813303</td>
      <td>-0.075233</td>
      <td>-0.760479</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>1.677670</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>0.695681</td>
      <td>0.968487</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.286714</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>-0.700472</td>
      <td>-0.760479</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.286714</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>0.367466</td>
      <td>2.032467</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.286714</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>-0.604063</td>
      <td>-0.760479</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>-0.734498</td>
      <td>0.436498</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.039245</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>-0.662192</td>
      <td>-0.361487</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.039245</td>
      <td>-1.491914</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>0.124676</td>
      <td>-1.344000</td>
      <td>-0.705079</td>
      <td>0.436498</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.955847</td>
      <td>0.813303</td>
      <td>-0.300305</td>
      <td>0.436498</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.286714</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.955847</td>
      <td>-1.344000</td>
      <td>1.701591</td>
      <td>0.968487</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-1.218348</td>
      <td>-1.491914</td>
      <td>1.677670</td>
      <td>-2.016956</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>3.299067</td>
      <td>0.436498</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>1.787019</td>
      <td>0.813303</td>
      <td>0.056265</td>
      <td>0.436498</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>-0.403094</td>
      <td>-1.159472</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>1.749535</td>
      <td>1.787019</td>
      <td>-1.344000</td>
      <td>-0.221264</td>
      <td>-1.558464</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>-0.365168</td>
      <td>-1.026474</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.124676</td>
      <td>-0.265348</td>
      <td>-0.520060</td>
      <td>-0.760479</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.955847</td>
      <td>0.813303</td>
      <td>-0.426132</td>
      <td>-1.026474</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>-0.672471</td>
      <td>-1.558464</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>-1.008483</td>
      <td>-1.558464</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-1.218348</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>1.749535</td>
      <td>2.618191</td>
      <td>1.891955</td>
      <td>-1.014508</td>
      <td>-0.760479</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>-0.303495</td>
      <td>-1.425467</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.465817</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>1.263499</td>
      <td>2.298462</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>970</th>
      <td>0.286714</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.124676</td>
      <td>-0.265348</td>
      <td>-0.622848</td>
      <td>-0.361487</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>6.667424</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>971</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>1.460924</td>
      <td>0.436498</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>972</th>
      <td>1.039245</td>
      <td>-1.491914</td>
      <td>-2.914492</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>-0.736625</td>
      <td>0.436498</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>973</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>1.426898</td>
      <td>2.298462</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>974</th>
      <td>1.039245</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>-0.156047</td>
      <td>0.968487</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>975</th>
      <td>-1.218348</td>
      <td>-1.491914</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>1.787019</td>
      <td>1.891955</td>
      <td>-0.713586</td>
      <td>0.436498</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>976</th>
      <td>-0.465817</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>-0.892580</td>
      <td>-1.558464</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>977</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.955847</td>
      <td>-0.265348</td>
      <td>-0.299242</td>
      <td>-0.095492</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>978</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>-0.259898</td>
      <td>0.436498</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>979</th>
      <td>0.286714</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>1.749535</td>
      <td>0.124676</td>
      <td>-0.265348</td>
      <td>-0.711459</td>
      <td>-0.361487</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>980</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>1.812886</td>
      <td>0.968487</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>981</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>1.677670</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>0.557448</td>
      <td>2.032467</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>982</th>
      <td>1.039245</td>
      <td>-1.491914</td>
      <td>1.677670</td>
      <td>-0.133710</td>
      <td>0.124676</td>
      <td>1.891955</td>
      <td>-0.123438</td>
      <td>0.170503</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>983</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>1.757239</td>
      <td>1.234482</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>984</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>-0.440665</td>
      <td>0.436498</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>985</th>
      <td>0.286714</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>1.749535</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>-0.651558</td>
      <td>-0.361487</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>986</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>1.891955</td>
      <td>1.069619</td>
      <td>1.633475</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>987</th>
      <td>-0.465817</td>
      <td>-1.491914</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.124676</td>
      <td>0.813303</td>
      <td>-0.660065</td>
      <td>-0.627482</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>988</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>1.677670</td>
      <td>-2.016956</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>1.172407</td>
      <td>0.436498</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>989</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-0.265348</td>
      <td>-0.541681</td>
      <td>0.436498</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>990</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>0.955847</td>
      <td>0.813303</td>
      <td>0.104115</td>
      <td>-0.760479</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>3.993639</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>991</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>0.124676</td>
      <td>0.813303</td>
      <td>-0.603354</td>
      <td>-0.361487</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>992</th>
      <td>0.286714</td>
      <td>0.670280</td>
      <td>-1.383771</td>
      <td>1.749535</td>
      <td>0.955847</td>
      <td>-1.344000</td>
      <td>-0.473273</td>
      <td>-0.095492</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>993</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>1.677670</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>0.243766</td>
      <td>1.234482</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>994</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.955847</td>
      <td>0.813303</td>
      <td>-0.312356</td>
      <td>-0.760479</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1.039245</td>
      <td>-1.491914</td>
      <td>-1.383771</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>-0.544162</td>
      <td>-0.760479</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>2.127172</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>996</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>1.677670</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>0.207612</td>
      <td>0.968487</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>997</th>
      <td>-1.218348</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>-0.706496</td>
      <td>0.813303</td>
      <td>-0.874503</td>
      <td>-0.760479</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0.286714</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-2.016956</td>
      <td>-0.706496</td>
      <td>-1.344000</td>
      <td>-0.505528</td>
      <td>1.766472</td>
      <td>-0.712949</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>1.603567</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1.039245</td>
      <td>0.670280</td>
      <td>0.146949</td>
      <td>-0.133710</td>
      <td>0.124676</td>
      <td>-0.265348</td>
      <td>0.462457</td>
      <td>1.766472</td>
      <td>1.402626</td>
      <td>-0.110208</td>
      <td>-0.250398</td>
      <td>-0.470108</td>
      <td>-0.623610</td>
      <td>-0.149983</td>
      <td>-0.110208</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 15 columns</p>
</div>



# 5) Building Models

Now our data is in perfect form to build a model so we are going to import our requied libraries from the Scikit learn.


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
```

Spliting our data into training data set and testing data set of 75-25 combo.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
```

we are going to build a Logistic Regression model on the dataset.Logistics Regression is the better option for the classification problems with two possible outcomes.Here also our target varible consists of two posible outcomes os Good and Bad.


```python
logreg = LogisticRegression()
logreg.fit(X_train ,y_train)
y_pred = logreg.predict(X_test)
```

Building ROC curve


```python
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--1')
plt.plot(fpr,tpr,label = 'Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Curve')
```




    Text(0.5,1,'Logistic Regression Curve')




![png](output_44_1.png)


Calculating the ROC score to find the performance of the model.If area under ROC curve is high,then model is good otherwise termed it as bad


```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_prob)
```




    0.7556960049937579



Implementing Decision Tree model and finding its scores


```python
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print(tree.score(X_train,y_train))
print(tree.score(X_test,y_test))
```

    0.764
    0.744
    

Implementing Random forest model and finding its scores


```python
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1,
                oob_score=False, random_state=2, verbose=0, warm_start=False)




```python
print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))
```

    0.9653333333333334
    0.66
    

Implementing Gradient boost classifier model and finding its scores


```python
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))
```

    0.748
    0.728
    

From the above models and their performance scores we can go with Logstic Regression Model
