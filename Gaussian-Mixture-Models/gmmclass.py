import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

class GaussianMixtureClustering():
    def __init__(self, k):
        self.k = k
        self.pi = np.ones(self.k) / self.k 
        self.classes = None
        self.costs = None
        self.means = None
        self.covs = None
    
    def fit(self, X, max_iter, regularizer = 1e-2):
        N, d = X.shape
        self.means = np.zeros((self.k, d))
        self.covs = np.zeros((self.k, d, d))
        self.classes = np.zeros((N, self.k))
  
        # initializing means to random and covariances 1
        for k in range(self.k):
            self.means[k] = X[np.random.choice(N)]
            self.covs[k] = np.eye(d)

        self.costs = np.zeros(max_iter)
        weighted_pdfs = np.zeros((N, self.k)) #store the PDF value of sample n and Gaussian k
        for i in range(max_iter):
            # EM algorithm 
            # step 1: assigns classes based on prior distribution
            for k in range(self.k):
                weighted_pdfs[:,k] = self.pi[k]*multivariate_normal.pdf(X, self.means[k], self.covs[k])

            self.classes = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

            # step 2: calculates posteriror means and covariances based on classes assigned
            for k in range(self.k):
                Nk = self.classes[:,k].sum()
                self.pi[k] = Nk / N
                self.means[k] = self.classes[:,k].dot(X) / Nk
                self.covs[k] = np.sum(self.classes[n,k]*np.outer(X[n] - self.means[k], X[n] - self.means[k]) for n in range(N)) / Nk + np.eye( d ) * regularizer


            self.costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()
            if i > 0:
                if np.abs(self.costs[i] - self.costs[i-1]) < 0.1:
                    break


    def predict(self):
        return self.classes.argmax(axis=1)
                
    def plot(self, X):
        plt.plot(self.costs)
        plt.title("Costs")
        plt.show()

        random_colors = np.random.random((self.k, 3))
        plt.scatter(X[:,0], X[:,1], c = self.classes.argmax(axis=1))
        plt.show()

        print("pi:", self.pi)
        print("means:", self.means)
        print("covariances:", self.covs)


def main():
    X = pd.read_csv('data.txt', header=None).as_matrix()
    
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    
    for k in (2,4,8,10):
        model = GaussianMixtureClassifier(k)
        model.fit( X, max_iter=1000, regularizer = 0 )
        model.plot( X )

if __name__ == '__main__':
    main()
