
import numpy as np


class LinearRegressionOLS:
    
    """
    This class fits a linear regression using Normal Equation (ordinary least squares).
    The design matrix must be normalized in order to work.
    """
    
    def reshape_x(self, X):
        """
        Verifies if X object is a numpy array. It can be a pandas object too.
        """
        if not isinstance(X, np.ndarray):
            X = X.values
            
        # Reshaping the design matrix, if necessary
        if len(X.shape) == 1:
            Xn = np.reshape(X, (X.shape[0], 1))
        else:
            Xn = X.copy()
            
        # adding a new column of ones in Xn to calculate the intercept
        m = Xn.shape[0]
        Xn = np.insert(Xn, 0, np.ones(m), axis = 1)
        
        return Xn
    
    def reshape_y(self, y):
        """
        Verifies if y is a numpy array.
        """
        if not isinstance(y, np.ndarray):
            y = y.values
            
        y = np.reshape(y, (y.shape[0],1))
        
        return y
    
    def fit(self, X, y):      
        """
        Fits the model.
        """
        # Reshapes X
        Xn = self.reshape_x(X)
        
        # Reshapes y
        y = self.reshape_y(y)
        
        # Fit with Normal Equation
        inverse_matrix = np.linalg.inv(np.dot(Xn.T, Xn))
        
        self.coef = np.dot(np.dot(inverse_matrix, Xn.T), y)
            
    def predict(self, X):
        """
        Makes predictions.
        """
        # Reshaping the design matrix
        Xn = self.reshape_x(X)
        
        y_pred = np.dot(Xn, self.coef)
        
        return y_pred.flatten()
    
    def score(self, y_pred, y_true):
        """
        Estimates the score.
        """
        return np.sum(((y_true - y_pred)**2))/y_true.shape[0]


class LinearRegressionGD(LinearRegressionOLS):
    
    """
    This class fits a linear regression using gradient descent as the optimization algorithm.
    The design matrix must be normalized in order to work.
    """
    
    def __init__(self, alpha = 0.01, iterations = 1000, tolerance = 0.00001):
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
        self.coef = np.random.randn(n, 1)
            
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

