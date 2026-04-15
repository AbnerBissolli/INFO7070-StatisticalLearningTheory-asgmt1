import numpy as np

class SmoothKernels():
    def cosine(u):
        return (np.pi/4) * np.cos((np.pi * u) / 2) * (np.abs(u) <= 1)
        
    def epanechnikov(u):
        return (3/4) * (1 - u**2) * (np.abs(u) <= 1)
    
    def exponential(u):
        return np.exp(-np.abs(u))/2
    
    def gaussian(u):
        return np.exp(-(u**2)/2)
    
    def triangular(u):
        return (1 - np.abs(u)) * (np.abs(u) <= 1)
    
    def uniform(u):
        return (np.abs(u) <= 1)/2

class KernelKNN:
    def __init__(self, k=5, h=1.0, kernel=None):
        self.k = k
        self.h = h
        self.kernel = kernel
   
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
        
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            idx = np.argsort(distances)[:self.k]
            k_distances = distances[idx]
            k_labels = self.y_train[idx]
            
            # K(distance / h)
            u = k_distances / self.h
            weights = self.kernel(u)
            
            if np.sum(weights) > 0:
                pred = np.sum(weights * k_labels) / np.sum(weights)
            else:
                pred = np.mean(k_labels)
            
            predictions.append(pred)
        return np.array(predictions)


class NadarayaWatson:
    def __init__(self, h=1.0, kernel=None):
        self.h = h 
        self.kernel = kernel
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Normalize by bandwidth
            u = distances / self.h
            weights = self.kernel(u)
            
            # All points
            if np.sum(weights) > 0:
                pred = np.sum(weights * self.y_train) / np.sum(weights)
            else:
                pred = np.mean(self.y_train)
            
            predictions.append(pred)
        
        return np.array(predictions)


        