# -*- coding: utf-8 -*-
"""

Author: NeetKing

"""

# Importing the libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Importing the dataset 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) 

# Fitting Simple Linear Regression to the Training set 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

# Predicting the Test set results 

y_pred = regressor.predict(X_test) 

# Visualising the Training set results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results 

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 

regressor.predict([[10.5]]) 

#Calculate the performance
# r2 linear
from sklearn.metrics import r2_score  
coefficient_of_dermination = r2_score(y_test, y_pred) 
print(coefficient_of_dermination)


 
