import numpy as np

class LogisticRegression:
    
    '''Regularized Logistic Regression with Gradient Descent. Regularization Ridge ('l2') is implemented.
    It is advisable to normalize the data.
    
    alpha: learning rate. It should be positive.
    iterations: number of iterations. Positive integer.
    tolerance: stop iterations when tolerance in estimated coefficients are smaller than the tolerance. Positive float.
    penalty: apply regularization. Boolean.
    C: regularization term. Positive.'''
    
    def __init__(self, alpha=0.1, iterations=1000, tolerance = 0.0001, penalty = True, C=1):
        self.alpha = alpha
        self.iterations = iterations
        self.tol = tolerance
        self.penalty = penalty
        self.C = C
    
    def reshape_x(self, X):
        # verify if X object is an array numpy. It can be a pandas object too.
        if not isinstance(X, np.ndarray):
            X = X.values
        
        # Reshaping the design matrix, if necessary
        if len(X.shape) == 1:
            Xn = np.reshape(X, (X.shape[0],1))
        else:
            Xn = X.copy()
            
        # adding a new column of ones in Xn to calculate the intercept
        m = Xn.shape[0]
        Xn = np.insert(Xn, 0, np.ones(m), axis = 1)
        return Xn
    
    def reshape_y(self,y):
        # verifying if y is an array numpy
        if not isinstance(y, np.ndarray):
            y = y.values
        y = np.reshape(y, (y.shape[0],1))
        return y
    
    
    def sigmoid(self, X):
        
        '''This is our hypothesis h = sigmoid(z); z = X*coef'''
        
        return 1.0/(1.0 + np.exp(-np.dot(X,self.coef)))
    
    def cost_function(self,X,y,m):
        cost = np.dot(y.T,np.log(self.sigmoid(X))) + np.dot((1-y).T,np.log(1- self.sigmoid(X)))
        if self.penalty:
            J = - (cost + self.C*np.dot(self.coef.T, self.coef)/2) / m
        else:
            J = - cost / m
        return J
    
    def gradient(self,X, y,m):
        return np.dot(X.T, (self.sigmoid(X) - y)) / m
    
    def fit(self, X, y):
        
        # Reshape X and y to suitable dimensions
        Xn = self.reshape_x(X)
        yn = self.reshape_y(y)
        m = X.shape[0]
        
        # Inicializing the coefficients
        n = Xn.shape[1]
        self.coef = np.random.randn(n,1)
        
        # Vector to save the cost values
        J = np.zeros(self.iterations)
        
        for i in range(self.iterations):
            
            coef_lastStep = self.coef.copy()
            
            # Cost Function
            J[i] = self.cost_function(Xn,yn,m)
            
            # Gradient Descent
            gradient = self.gradient(Xn,yn,m)
            
            if self.penalty:
                
                # update coefficients
                #intercept is not penalized
                self.coef[0] = self.coef[0] - self.alpha*gradient[0]
                
                self.coef[1:] = self.coef[1:]*(1 - self.alpha*self.C/m) - self.alpha*gradient[1:]
            
            else:
                # update coefficients
                self.coef = self.coef - self.alpha*gradient
            
            if all(np.abs(self.coef - coef_lastStep) < self.tol):
                print('Converged in iterations: ',i)
                break
        return J
    
    def predict_proba(self,X):
        Xn = self.reshape_x(X)
        return self.sigmoid(Xn).flatten()
    
    
    def predict(self, X):
        
        '''Setting the decision boundary as:
            y = 1 if h > 0.5 or X*coef > 0
            y = 0 if h <= 0.5 or X*coef <=0 '''
        
        Xn = self.reshape_x(X)
        y_pred = self.sigmoid(Xn).flatten()
        # Threshold = 0.5
        return np.round(y_pred > 0.5)
    
    def score(self,y_pred, y):
        return np.mean(y_pred == y)

