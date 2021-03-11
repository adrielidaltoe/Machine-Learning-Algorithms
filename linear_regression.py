#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class LinearRegressionOLS:
    
    '''This class fits a linear regression using Normal Equation (ordinary least squares).
    Normalize the design matrix.'''
    
    def reshape_x(self,X):
        # Reshaping the design matrix, if necessary
        if len(X.shape) == 1:
            # verify if X object is an array numpy. It can be a pandas object too.
            if type(X) != 'numpy.ndarray':
                X = X.values
            Xn = np.reshape(X, (X.shape[0],1))
        else:
            Xn = X.copy()
            
        # adding a new column of ones in Xn to calculate the intercept
        m = Xn.shape[0]
        Xn = np.insert(Xn, 0, np.ones(m), axis = 1)
        
        return Xn
    
    def reshape_y(self, y):
        # verifying if y is an array numpy
        if type(y) != 'numpy.ndarray':
            y = y.values
        y = np.reshape(y, (y.shape[0],1))
        return y
    
    def fit(self, X, y):
                
        # Reshaping the design matrix
        Xn = self.reshape_x(X)
        
        # Reshaping y
        y = self.reshape_y(y)
        
        # Fit with Normal Equation
        inverse_matrix = np.linalg.inv(np.dot(Xn.T,Xn))
        self.coef = np.dot(np.dot(inverse_matrix,Xn.T), y)
            
    def predict(self, X):
        
        # Reshaping the design matrix, if necessary
        Xn = self.reshape_x(X)
        y_pred = np.dot(Xn,self.coef)
        return y_pred.flatten()
    
    def score(self, y_pred, y_true):
        return np.sum(((y_true - y_pred)**2))/y_true.shape[0]


# In[61]:


class LinearRegressionGD(LinearRegressionOLS):
    
    '''This class fits a linear regression using gradient descent as the optimization algorithm.
    Normalize the design matrix.'''
    
    def __init__(self, iterations = 1000, alpha = 0.01, tolerance = 0.00001):
        self.alpha = alpha
        self.iterations = iterations
        self.tol = tolerance
        
    def fit(self, X, y):
                
        # Reshaping the design matrix
        Xn = self.reshape_x(X)
        
        # Reshaping y
        y = self.reshape_y(y)
        
        # initializing the parameters. Xn is now a matrix [m x n+1]
        n = Xn.shape[1]
        self.coef = np.random.randn(n,1)
            
        # Size of training set
        m = Xn.shape[0]
            
        # Initializing cost vector
        J = np.zeros(self.iterations)
        
        for i in range(self.iterations):
            coef_lastStep = self.coef.copy()
            # Cost function
            J[i] = 1/(2*m) *np.sum(((np.dot(Xn, self.coef) - y) ** 2))
        
            # Gradient Descend
            gradient = np.dot(Xn.T, (np.dot(Xn, self.coef) - y))/m
            
            # update parameters
            self.coef = self.coef - self.alpha*gradient
            
            if all(np.abs(self.coef - coef_lastStep) < self.tol):
                print('Converged in iteration', i)
                break
            
        return J

