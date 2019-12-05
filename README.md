# ML - Linear Regression

## Overview

Linear regression is a part of supervised machine learning. Linear regression is the best fit line for the given data point. It refers to a linear relationship (straight line) between the independent variables and the dependent variables.
 
Simple linear regression involves two variables where an independent value (X column) and a dependent value (Y column). In this article, we try to predict an employee's salary based on salary. We implement python code using the Scikit-Learn machine learning library. All code is executed under Python.

###  Importing the libraries 

* numpy `import numpy as np` 
* matplotlib `import matplotlib.pyplot as plt`
* pandas `import pandas as pd`

### Importing the dataset

```python
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
```
![](/imgs/import-dataset.jpg?raw=true)

### Split data

first we split the dataset into training and test data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```
now, we can train a classifier to try to predict the category of a new critic
`Scikit-learn`, our preferred framework, provides help in choosing the algorithm:

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
Test the model and display `y_pred`, `y_test`
```python
y_pred = regressor.predict(X_test)
```

### View the result

```python
# Visualising the Training set results 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```
![](/imgs/view-result.jpg?raw=true)
```python
# Visualising the Test set results 
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 
```
![](/imgs/view-result2.jpg?raw=true)

### Prediction of new salary
Once the supervised classification model is created, we proceed to the prediction:
```python
regressor.predict([[5.6]]) 
```
![](/imgs/predection.jpg?raw=true)

Enjoy the codes.
