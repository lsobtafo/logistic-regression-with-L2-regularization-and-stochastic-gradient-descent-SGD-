########## >>>>>> Put your full name and 6-digit EWU ID here. 

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent




import numpy as np
import math
import sys
sys.path.append("..")

from code_misc.utils import MyUtils



class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1



    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''

        # remove the pass statement and fill in the code. 

        #vs = LogisticRegresssion._v_sigmoid()
        self.degree = degree
        X = MyUtils.z_transform(X, degree=self.degree)
        #np.random.seed()
        n, d=X.shape
        
        #bias case
        
        
        #initialize self.w
        #self.w =np.random.rand(d+1,1)
        #self.w =(self.w * 0 - 1)/math.sqrt(d)
        self.w = np.zeros((d+1,1))
        X = np.insert(X, 0, 1, axis=1)
        #Code from class
        #init self.w same as you did for linear regresssion
        if(SGD):
            num_of_batches = math.ceil(n/mini_batch_size)
            for i in range(iterations):
                batchNumber = i % num_of_batches
                start = batchNumber * mini_batch_size
                end = (batchNumber +1) * mini_batch_size
                X_prime = X[start:end]
                Y_prime = y[start:end]
                n_prime, d_prime = X_prime.shape
                self.GDcalculation(X_prime, Y_prime, lam, eta,n_prime)
        else:
            while iterations > 0:
                self.GDcalculation(X,y,lam,eta,n)
                iterations -= 1
    def GDcalculation(self,X,y,lam,eta,n):
        s=y*(X@self.w)
        tp = 1 -(2*lam*eta)/n
        self.w =tp*self.w+(eta/n*(X.T@(y*LogisticRegression._v_sigmoid(-s))))






                                                                 
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
        #X -> Z(X) -> add bias column --> X
        #a = X @self.w
        # remove the pass statement and fill in the code. 
        X = MyUtils.z_transform(X, degree=self.degree)
        X = np.insert(X, 0, 1, axis=1)
        pred = X @ self.w
        return LogisticRegression._v_sigmoid(pred)                                                          
                                                                   
                                                                   
                                                                
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''
        #return np.sum(np.sign((np.sign(X@self.w)-0.1)) != y)
        
        
        # remove the pass statement and fill in the code.         
        #n, d = X.shape
        #Z = MyUtils.Z(X, self.degree)
        #X = np.concatenate((np.ones((n, 1)), Z), axis=1)
        X = MyUtils.z_transform(X, degree=self.degree)
        X = np.insert(X, 0, 1, axis=1)
        pred = np.sign(X@self.w) 
        pred = np.sign(pred - 0.1)
        #MSE = np.sign(pred - 0.1)                                                       MSE = np.sum(pred !=y)
        MSE = np.sum(pred != y)
        return MSE                                                          
                                                                   
    
    
    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
            
        # Hint: use the np.vectorize API

        # remove the pass statement and fill in the code.         
        #return np.vectorize(LogisticRegression._sigmoid)
        
                           
        #vs = np.vectorize(LogisticRegressoin._sigmoid)
        #return vs(s)
        vs = np.vectorize(LogisticRegression._sigmoid)
        return vs(s)                                                         
       
    
        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''

        # remove the pass statement and fill in the code.
        return 1 / (1 + np.exp(-s))
