# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:32:20 2019

@author: Benjamin B
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values # Need range to make x a matrix, not a vector
y = dataset.iloc[:, 2].values

# Splitting the dataset into Training and Testing sets
""" Do not need since only 10 observations and need VERY accurate prediction**

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Transform tool (x --> x^2, etc)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# Visualizing the Linear Regression results
plt.scatter(x, y, color = 'red') # Real data POINTS
plt.plot(x, lin_reg.predict(x), color = 'green') # Prediction LINE
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red') # Real data POINTS
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = 'blue') # Prediction LINE
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
